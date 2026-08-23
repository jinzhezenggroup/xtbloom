! xtbloom unrestricted oracle trace recorder (issue #51).
!
! Builds a pinned GFN2 calculator, optionally applies the oracle-only
! shell-monopole PCEM container, runs xtb_singlepoint with the observer sewn
! into the pinned tblite v2 patch, and streams every xtbloom-scc-trace-v2 field in
! a fixed line-oriented raw format on stdout.  generate_scc_corpus.py parses
! that stream and serializes the canonical versioned JSON document via the
! canonical writer so no field semantics live in Fortran.
!
! CLI:
!   scc_trace_recorder --nat N --atoms z1..zN --xyz x1 y1 z1 ... \
!     --charge C --unpaired U --temperature K --mixer-memory M \
!     --mixer-damping D --max-iter T \
!     [--pcem-xyz x1 y1 z1 ... --pcem-q q1 ... --pcem-gamma g1 ...]
!
! The raw stream layout is documented in generate_scc_corpus.py and is covered
! by offline fixtures in tests/oracle/test_scc_corpus_io.py.
module xtbloom_tblite_scc_trace_recorder_v2
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use mctc_env, only : wp, error_type
   use mctc_io, only : structure_type, new
   use tblite_basis_type, only : basis_type
   use tblite_container_cache, only : container_cache
   use tblite_container_type, only : container_type
   use tblite_context_type, only : context_type
   use tblite_data_spin, only : get_spin_constant
   use tblite_results, only : results_type
   use tblite_scf, only : potential_type, scf_observer, &
      & scf_observer_status_converged, scf_observer_status_max_iterations, &
      & scf_observer_status_failed
   use tblite_scf_info, only : scf_info, shell_resolved
   use tblite_scf_mixer_input, only : scf_version
   use tblite_spin, only : spin_polarization, new_spin_polarization
   use tblite_wavefunction, only : wavefunction_type, new_wavefunction
   use tblite_xtb_calculator, only : xtb_calculator
   use tblite_xtb_gfn2, only : new_gfn2_calculator
   use tblite_xtb_singlepoint, only : xtb_singlepoint
   implicit none
   private

   public :: play

   integer, parameter :: max_observed_iterations = 512

   type, extends(container_type) :: shell_monopole_pcem
      integer :: nsh = 0
      integer :: nat = 0
      integer :: npun = 0
      integer, allocatable :: sh2at(:)
      real(wp), allocatable :: xyz(:, :)
      real(wp), allocatable :: gamma_shell(:)
      real(wp), allocatable :: pc_xyz(:, :)
      real(wp), allocatable :: pc_charge(:)
      real(wp), allocatable :: pc_gamma(:)
      real(wp), allocatable :: vpc(:)
   contains
      procedure :: variable_info => pcem_variable_info
      procedure :: get_energy => pcem_get_energy
      procedure :: get_potential => pcem_get_potential
      procedure :: get_potential_gradient => pcem_get_potential_gradient
      procedure :: get_gradient => pcem_get_gradient
      procedure :: get_engrad => pcem_get_engrad
   end type shell_monopole_pcem

   ! One slab per completed iteration.  The pre-solve payload (effective
   ! Hamiltonian and mixed q/d/Q) is captured by before_solve, and the
   ! post-solve payload (eigenvalues, occupations, density, raw q/d/Q, energy,
   ! residual RMS, convergence) by after_iteration, both indexed by the same
   ! one-based attempt number so the stream order is unambiguous.
   type, extends(scf_observer) :: recording_observer
      integer :: count = 0
      integer :: terminal_status = 0
      real(wp), allocatable :: assembled_hamiltonian(:, :, :, :)
      real(wp), allocatable :: solver_hamiltonian_values(:, :, :, :)
      real(wp), allocatable :: mixed_qsh(:, :, :)
      real(wp), allocatable :: mixed_qat(:, :, :)
      real(wp), allocatable :: mixed_dpat(:, :, :, :)
      real(wp), allocatable :: mixed_qpat(:, :, :, :)
      real(wp), allocatable :: density(:, :, :, :)
      real(wp), allocatable :: raw_qsh(:, :, :)
      real(wp), allocatable :: raw_qat(:, :, :)
      real(wp), allocatable :: raw_dpat(:, :, :, :)
      real(wp), allocatable :: raw_qpat(:, :, :, :)
      real(wp), allocatable :: emo(:, :, :)
      real(wp), allocatable :: focc(:, :, :)
      real(wp), allocatable :: eelec_sum(:)
      real(wp), allocatable :: energy_delta(:)
      real(wp), allocatable :: pnorm(:)
      logical, allocatable :: econv(:), pconv(:), tconv(:), conv(:)
   contains
      procedure :: before_solve => capture_before_solve
      procedure :: solver_hamiltonian => capture_solver_hamiltonian
      procedure :: after_iteration => capture_after_iteration
      procedure :: finished => capture_finished
   end type recording_observer

contains

subroutine collect_spin_constants(wll, mol, bas)
   real(wp), allocatable, intent(out) :: wll(:, :, :)
   type(structure_type), intent(in) :: mol
   type(basis_type), intent(in) :: bas

   integer :: izp, ish, jsh, il, jl

   allocate(wll(bas%nsh, bas%nsh, mol%nid), source=0.0_wp)
   do izp = 1, mol%nid
      do ish = 1, bas%nsh_id(izp)
         il = bas%cgto(ish, izp)%ang
         do jsh = 1, bas%nsh_id(izp)
            jl = bas%cgto(jsh, izp)%ang
            wll(jsh, ish, izp) = get_spin_constant(jl, il, mol%num(izp))
         end do
      end do
   end do
end subroutine collect_spin_constants

subroutine compute_vpc(self, mol)
   class(shell_monopole_pcem), intent(inout) :: self
   type(structure_type), intent(in) :: mol
   integer :: sh, pu
   real(wp) :: dx, dy, dz, screening, distance

   if (allocated(self%vpc)) deallocate(self%vpc)
   allocate(self%vpc(self%nsh), source=0.0_wp)
   do sh = 1, self%nsh
      do pu = 1, self%npun
         dx = mol%xyz(1, self%sh2at(sh)) - self%pc_xyz(1, pu)
         dy = mol%xyz(2, self%sh2at(sh)) - self%pc_xyz(2, pu)
         dz = mol%xyz(3, self%sh2at(sh)) - self%pc_xyz(3, pu)
         screening = 2.0_wp / (self%gamma_shell(sh) + self%pc_gamma(pu))
         distance = sqrt(dx*dx + dy*dy + dz*dz + screening*screening)
         self%vpc(sh) = self%vpc(sh) + self%pc_charge(pu) / distance
      end do
   end do
end subroutine compute_vpc

pure function pcem_variable_info(self) result(info)
   class(shell_monopole_pcem), intent(in) :: self
   type(scf_info) :: info
   info = scf_info(charge=shell_resolved)
end function pcem_variable_info

subroutine pcem_get_energy(self, mol, cache, wfn, energies)
   class(shell_monopole_pcem), intent(in) :: self
   type(structure_type), intent(in) :: mol
   type(container_cache), intent(inout) :: cache
   type(wavefunction_type), intent(in) :: wfn
   real(wp), intent(inout) :: energies(:)
   integer :: sh, at

   do sh = 1, self%nsh
      at = self%sh2at(sh)
      energies(at) = energies(at) + wfn%qsh(sh, 1) * self%vpc(sh)
   end do
end subroutine pcem_get_energy

subroutine pcem_get_potential(self, mol, cache, wfn, pot)
   class(shell_monopole_pcem), intent(in) :: self
   type(structure_type), intent(in) :: mol
   type(container_cache), intent(inout) :: cache
   type(wavefunction_type), intent(in) :: wfn
   type(potential_type), intent(inout) :: pot

   pot%vsh(:, 1) = pot%vsh(:, 1) + self%vpc
end subroutine pcem_get_potential

subroutine pcem_get_potential_gradient(self, mol, cache, wfn, pot)
   class(shell_monopole_pcem), intent(in) :: self
   type(structure_type), intent(in) :: mol
   type(container_cache), intent(inout) :: cache
   type(wavefunction_type), intent(in) :: wfn
   type(potential_type), intent(inout) :: pot
end subroutine pcem_get_potential_gradient

subroutine pcem_get_gradient(self, mol, cache, wfn, gradient, sigma)
   class(shell_monopole_pcem), intent(in) :: self
   type(structure_type), intent(in) :: mol
   type(container_cache), intent(inout) :: cache
   type(wavefunction_type), intent(in) :: wfn
   real(wp), contiguous, intent(inout) :: gradient(:, :)
   real(wp), contiguous, intent(inout) :: sigma(:, :)
end subroutine pcem_get_gradient

subroutine pcem_get_engrad(self, mol, cache, energies, gradient, sigma)
   class(shell_monopole_pcem), intent(in) :: self
   type(structure_type), intent(in) :: mol
   type(container_cache), intent(inout) :: cache
   real(wp), intent(inout) :: energies(:)
   real(wp), contiguous, intent(inout), optional :: gradient(:, :)
   real(wp), contiguous, intent(inout), optional :: sigma(:, :)
end subroutine pcem_get_engrad

subroutine capture_before_solve(self, iteration, wfn, pot)
   class(recording_observer), intent(inout) :: self
   integer, intent(in) :: iteration
   type(wavefunction_type), intent(in) :: wfn
   type(potential_type), intent(in) :: pot

   self%assembled_hamiltonian(:, :, :, iteration) = wfn%coeff
   self%mixed_qsh(:, :, iteration) = wfn%qsh
   self%mixed_qat(:, :, iteration) = wfn%qat
   self%mixed_dpat(:, :, :, iteration) = wfn%dpat
   self%mixed_qpat(:, :, :, iteration) = wfn%qpat
end subroutine capture_before_solve

subroutine capture_solver_hamiltonian(self, iteration, hamiltonian)
   class(recording_observer), intent(inout) :: self
   integer, intent(in) :: iteration
   real(wp), intent(in) :: hamiltonian(:, :, :)

   self%solver_hamiltonian_values(:, :, :, iteration) = hamiltonian
end subroutine capture_solver_hamiltonian

subroutine capture_after_iteration(self, iteration, wfn, eelec, elast, pnorm, &
      & econverged, pconverged, tconverged, converged)
   class(recording_observer), intent(inout) :: self
   integer, intent(in) :: iteration
   type(wavefunction_type), intent(in) :: wfn
   real(wp), intent(in) :: eelec(:)
   real(wp), intent(in) :: elast
   real(wp), intent(in) :: pnorm
   logical, intent(in) :: econverged
   logical, intent(in) :: pconverged
   logical, intent(in) :: tconverged
   logical, intent(in) :: converged

   self%count = iteration
   self%density(:, :, :, iteration) = wfn%density
   self%raw_qsh(:, :, iteration) = wfn%qsh
   self%raw_qat(:, :, iteration) = wfn%qat
   self%raw_dpat(:, :, :, iteration) = wfn%dpat
   self%raw_qpat(:, :, :, iteration) = wfn%qpat
   self%emo(:, :, iteration) = wfn%emo
   self%focc(:, :, iteration) = wfn%focc
   self%eelec_sum(iteration) = sum(eelec)
   self%energy_delta(iteration) = sum(eelec) - elast
   self%pnorm(iteration) = pnorm
   self%econv(iteration) = econverged
   self%pconv(iteration) = pconverged
   self%tconv(iteration) = tconverged
   self%conv(iteration) = converged
end subroutine capture_after_iteration

subroutine capture_finished(self, iterations, status)
   class(recording_observer), intent(inout) :: self
   integer, intent(in) :: iterations
   integer, intent(in) :: status

   self%terminal_status = status
end subroutine capture_finished

subroutine play(atomic_numbers, positions, molecular_charge, unpaired_electrons, &
      & temperature, mixer_memory, mixer_damping, maximum_iterations, &
      & pc_positions, pc_charges, pc_gammas)
   integer, intent(in) :: atomic_numbers(:)
   real(wp), intent(in) :: positions(:, :)
   real(wp), intent(in) :: molecular_charge
   integer, intent(in) :: unpaired_electrons
   real(wp), intent(in) :: temperature
   integer, intent(in) :: mixer_memory
   real(wp), intent(in) :: mixer_damping
   integer, intent(in) :: maximum_iterations
   real(wp), intent(in), optional :: pc_positions(:, :)
   real(wp), intent(in), optional :: pc_charges(:)
   real(wp), intent(in), optional :: pc_gammas(:)

   type(structure_type) :: mol
   type(xtb_calculator) :: calc
   type(context_type) :: ctx
   type(wavefunction_type) :: wfn
   type(results_type) :: results
   type(recording_observer) :: observer
   type(spin_polarization), allocatable :: spin_container
   type(shell_monopole_pcem), allocatable :: pcem
   class(container_type), allocatable :: interaction_slot
   type(container_cache) :: cache
   type(error_type), allocatable :: error
   real(wp), parameter :: accuracy = 1.0_wp
   real(wp), parameter :: kt_per_kelvin = 3.166808578545117e-6_wp
   real(wp) :: energy, kt
   real(wp), allocatable :: saved_vpc(:), wll(:, :, :)
   integer :: nat, nsh, nao, nspin, iat, ish, iiter, row, col, ip, maxn, spin

   nat = size(atomic_numbers)
   call new(mol, atomic_numbers, positions, charge=molecular_charge, uhf=unpaired_electrons)
   call new_gfn2_calculator(calc, mol, error)
   if (allocated(error)) error stop "GFN2 calculator construction failed"

   calc%max_iter = maximum_iterations
   calc%mixer_input%memory = mixer_memory
   calc%mixer_input%damping = mixer_damping
   calc%mixer_input%scf = scf_version%broyden
   calc%save_integrals = .true.

   nsh = calc%bas%nsh
   nao = calc%bas%nao
   maxn = max_observed_iterations
   nspin = 2

   allocate(observer%assembled_hamiltonian(nao, nao, nspin, maxn))
   allocate(observer%solver_hamiltonian_values(nao, nao, nspin, maxn))
   allocate(observer%mixed_qsh(nsh, nspin, maxn), observer%mixed_qat(nat, nspin, maxn))
   allocate(observer%mixed_dpat(3, nat, nspin, maxn))
   allocate(observer%mixed_qpat(6, nat, nspin, maxn))
   allocate(observer%density(nao, nao, nspin, maxn))
   allocate(observer%raw_qsh(nsh, nspin, maxn), observer%raw_qat(nat, nspin, maxn))
   allocate(observer%raw_dpat(3, nat, nspin, maxn))
   allocate(observer%raw_qpat(6, nat, nspin, maxn))
   allocate(observer%emo(nao, nspin, maxn))
   allocate(observer%focc(nao, max(2, nspin), maxn))
   allocate(observer%eelec_sum(maxn), observer%energy_delta(maxn), observer%pnorm(maxn))
   allocate(observer%econv(maxn), observer%pconv(maxn), observer%tconv(maxn), observer%conv(maxn))

   ! A two-channel wavefunction only selects the unrestricted orbital layout.
   ! The GFN2 spin-polarization container supplies the independent
   ! magnetization potential and its 1/2 m^T W m energy, exactly as tblite's
   ! production CLI does for a spin-polarized calculation.
   allocate(spin_container)
   call collect_spin_constants(wll, mol, calc%bas)
   call new_spin_polarization(spin_container, mol, wll, calc%bas%nsh_id)
   call move_alloc(spin_container, interaction_slot)
   call calc%push_back(interaction_slot)

   if (present(pc_positions)) then
      allocate(pcem)
      pcem%nsh = nsh
      pcem%nat = nat
      pcem%npun = size(pc_charges)
      allocate(pcem%sh2at(nsh))
      do ish = 1, nsh
         pcem%sh2at(ish) = calc%bas%sh2at(ish)
      end do
      allocate(pcem%xyz(3, nat), source=positions)
      allocate(pcem%gamma_shell(nsh))
      call collect_shell_hardness(calc, mol, pcem%gamma_shell)
      allocate(pcem%pc_xyz(3, pcem%npun), source=pc_positions)
      allocate(pcem%pc_charge(pcem%npun), source=pc_charges)
      allocate(pcem%pc_gamma(pcem%npun), source=pc_gammas)
      pcem%label = "shell-monopole PCEM"
      call compute_vpc(pcem, mol)
      saved_vpc = pcem%vpc
      call move_alloc(pcem, interaction_slot)
      call calc%push_back(interaction_slot)
   end if

   kt = temperature * kt_per_kelvin
   call new_wavefunction(wfn, mol%nat, nsh, nao, nspin, kt)
   energy = 0.0_wp
   call xtb_singlepoint(ctx, mol, calc, wfn, accuracy, energy, verbosity=0, &
      & results=results, observer=observer)
   if (ctx%failed()) then
      write(*, '(a)') "SETUP_FAILURE"
      write(*, '(a,i0)') "terminal_status ", observer%terminal_status
      return
   end if

   write(*, '(a,i0,a,i0,a,i0,a,i0,a,i0)') "nat ", nat, " nsh ", nsh, " nao ", nao, &
      & " niterations ", observer%count, " terminal ", observer%terminal_status
   write(*, '(a)') "atomic_numbers"
   do iat = 1, nat
      write(*, '(i0)') mol%num(mol%id(iat))
   end do
   write(*, '(a)') "positions"
   do iat = 1, nat
      write(*, '(es26.17e3)') mol%xyz(1, iat)
      write(*, '(es26.17e3)') mol%xyz(2, iat)
      write(*, '(es26.17e3)') mol%xyz(3, iat)
   end do
   write(*, '(a,es26.17e3)') "molecular_charge ", molecular_charge
   write(*, '(a,i0)') "unpaired_electrons ", unpaired_electrons
   write(*, '(a,i0)') "spin_channels ", nspin
   write(*, '(a,es26.17e3)') "temperature ", temperature
   if (present(pc_positions)) then
      write(*, '(a,i0)') "n_point_charges ", size(pc_charges)
      do ip = 1, size(pc_charges)
         write(*, '(5(es26.17e3,1x))') pc_positions(1, ip), pc_positions(2, ip), &
            & pc_positions(3, ip), pc_charges(ip), pc_gammas(ip)
      end do
   else
      write(*, '(a,i0)') "n_point_charges 0"
   end if
   write(*, '(a)') "atom_to_shell_count"
   do iat = 1, nat
      write(*, '(i0)') calc%bas%nsh_at(iat)
   end do

   if (allocated(results%overlap)) then
      write(*, '(a)') "overlap"
      do col = 1, nao
         do row = 1, nao
            write(*, '(es26.17e3)') results%overlap(row, col)
         end do
      end do
   end if
   if (allocated(results%hamiltonian)) then
      write(*, '(a)') "core_hamiltonian"
      do col = 1, nao
         do row = 1, nao
            write(*, '(es26.17e3)') results%hamiltonian(row, col)
         end do
      end do
   end if

   do iiter = 1, observer%count
      write(*, '(a,i0)') "iteration ", iiter
      write(*, '(a)') "assembled_hamiltonian"
      do spin = 1, nspin
         do col = 1, nao
            do row = 1, nao
               write(*, '(es26.17e3)') observer%assembled_hamiltonian(row, col, spin, iiter)
            end do
         end do
      end do
      write(*, '(a)') "solver_hamiltonian"
      do spin = 1, nspin
         do col = 1, nao
            do row = 1, nao
               write(*, '(es26.17e3)') observer%solver_hamiltonian_values(row, col, spin, iiter)
            end do
         end do
      end do
      write(*, '(a)') "eigenvalues"
      do spin = 1, nspin
         do row = 1, nao
            write(*, '(es26.17e3)') observer%emo(row, spin, iiter)
         end do
      end do
      write(*, '(a)') "occupations"
      do spin = 1, 2
         do row = 1, nao
            write(*, '(es26.17e3)') observer%focc(row, spin, iiter)
         end do
      end do
      write(*, '(a)') "density"
      do spin = 1, nspin
         do col = 1, nao
            do row = 1, nao
               write(*, '(es26.17e3)') observer%density(row, col, spin, iiter)
            end do
         end do
      end do
      write(*, '(a)') "mixed_qsh"
      do spin = 1, nspin
         do ish = 1, nsh
            write(*, '(es26.17e3)') observer%mixed_qsh(ish, spin, iiter)
         end do
      end do
      write(*, '(a)') "mixed_qat"
      do spin = 1, nspin
         do iat = 1, nat
            write(*, '(es26.17e3)') observer%mixed_qat(iat, spin, iiter)
         end do
      end do
      write(*, '(a)') "mixed_dipoles"
      do spin = 1, nspin
         do iat = 1, nat
            write(*, '(es26.17e3)') observer%mixed_dpat(1, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%mixed_dpat(2, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%mixed_dpat(3, iat, spin, iiter)
         end do
      end do
      write(*, '(a)') "mixed_quadrupoles"
      do spin = 1, nspin
         do iat = 1, nat
            write(*, '(es26.17e3)') observer%mixed_qpat(1, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%mixed_qpat(2, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%mixed_qpat(3, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%mixed_qpat(4, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%mixed_qpat(5, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%mixed_qpat(6, iat, spin, iiter)
         end do
      end do
      write(*, '(a)') "raw_qsh"
      do spin = 1, nspin
         do ish = 1, nsh
            write(*, '(es26.17e3)') observer%raw_qsh(ish, spin, iiter)
         end do
      end do
      write(*, '(a)') "raw_qat"
      do spin = 1, nspin
         do iat = 1, nat
            write(*, '(es26.17e3)') observer%raw_qat(iat, spin, iiter)
         end do
      end do
      write(*, '(a)') "raw_dipoles"
      do spin = 1, nspin
         do iat = 1, nat
            write(*, '(es26.17e3)') observer%raw_dpat(1, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%raw_dpat(2, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%raw_dpat(3, iat, spin, iiter)
         end do
      end do
      write(*, '(a)') "raw_quadrupoles"
      do spin = 1, nspin
         do iat = 1, nat
            write(*, '(es26.17e3)') observer%raw_qpat(1, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%raw_qpat(2, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%raw_qpat(3, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%raw_qpat(4, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%raw_qpat(5, iat, spin, iiter)
            write(*, '(es26.17e3)') observer%raw_qpat(6, iat, spin, iiter)
         end do
      end do
      if (allocated(saved_vpc)) then
         write(*, '(a)') "point_charge_shell_potential"
         do ish = 1, nsh
            write(*, '(es26.17e3)') saved_vpc(ish)
         end do
         write(*, '(a)') "point_charge_energy"
         do ish = 1, nsh
            ! External point charges couple only to the charge channel; the
            ! magnetization channel has no independent electrostatic energy.
            write(*, '(es26.17e3)') observer%raw_qsh(ish, 1, iiter) * saved_vpc(ish)
         end do
      end if
      write(*, '(a)') "energy"
      write(*, '(es26.17e3)') observer%eelec_sum(iiter)
      write(*, '(a)') "energy_delta"
      write(*, '(es26.17e3)') observer%energy_delta(iiter)
      write(*, '(a)') "residual_rms"
      write(*, '(es26.17e3)') observer%pnorm(iiter)
      write(*, '(a)') "convergence"
      write(*, '(i0)') merge(1, 0, observer%econv(iiter))
      write(*, '(i0)') merge(1, 0, observer%pconv(iiter))
      write(*, '(i0)') merge(1, 0, observer%tconv(iiter))
      write(*, '(i0)') merge(1, 0, observer%conv(iiter))
   end do
end subroutine play

subroutine collect_shell_hardness(calc, mol, hardness)
   use tblite_xtb_gfn2, only : export_gfn2_param
   use tblite_param, only : param_record
   type(xtb_calculator), intent(in) :: calc
   type(structure_type), intent(in) :: mol
   real(wp), intent(out) :: hardness(:)
   type(param_record) :: param
   integer :: ish, at, izp, il, lsh, isp
   ! GFN2 shell hardness = element gam * shell Hubbard scale.  The per-shell
   ! product is exactly what xtbloom's generated parameter table exposes as
   ! shell_hardness for the external point-charge plan.  mol%id is the same
   ! species map used to construct calc%bas%cgto; atomic numbers are not valid
   ! indices into that species-major array.
   call export_gfn2_param(param)
   do ish = 1, size(hardness)
      at = calc%bas%sh2at(ish)
      isp = mol%id(at)
      izp = mol%num(isp)
      lsh = ish - calc%bas%ish_at(at)
      il = calc%bas%cgto(lsh, isp)%ang
      hardness(ish) = param%record(izp)%gam * param%record(izp)%lgam(il + 1)
   end do
end subroutine collect_shell_hardness

end module xtbloom_tblite_scc_trace_recorder_v2
