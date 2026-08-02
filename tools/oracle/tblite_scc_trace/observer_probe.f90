! Standalone integration probe for the gpuxtb tblite SCC observer patch.
module gpuxtb_tblite_observer_probe_type
   use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
   use mctc_env, only : wp
   use tblite_scf, only : potential_type, scf_observer
   use tblite_wavefunction, only : wavefunction_type
   implicit none
   private

   public :: recording_observer

   !> Deep-copying observer used only by the integration probe.
   type, extends(scf_observer) :: recording_observer
      integer :: before_count = 0
      integer :: after_count = 0
      integer :: first_before_iteration = 0
      integer :: first_after_iteration = 0
      integer :: last_after_iteration = 0
      integer :: finished_count = 0
      integer :: finished_iterations = -1
      integer :: finished_status = -1
      logical :: shapes_ok = .true.
      logical :: sequence_ok = .true.
      logical :: last_econverged = .false.
      logical :: last_pconverged = .false.
      logical :: last_tconverged = .false.
      logical :: last_converged = .false.
      real(wp) :: first_pnorm = 0.0_wp
      real(wp) :: last_pnorm = 0.0_wp
      real(wp) :: last_elast = 0.0_wp
      real(wp), allocatable :: first_hamiltonian(:, :, :)
      real(wp), allocatable :: first_mixed_qsh(:, :)
      real(wp), allocatable :: first_mixed_qat(:, :)
      real(wp), allocatable :: first_mixed_dpat(:, :, :)
      real(wp), allocatable :: first_mixed_qpat(:, :, :)
      real(wp), allocatable :: first_vsh(:, :)
      real(wp), allocatable :: first_vdp(:, :, :)
      real(wp), allocatable :: first_vqp(:, :, :)
      real(wp), allocatable :: first_coeff(:, :, :)
      real(wp), allocatable :: first_emo(:, :)
      real(wp), allocatable :: first_focc(:, :)
      real(wp), allocatable :: first_density(:, :, :)
      real(wp), allocatable :: first_raw_qsh(:, :)
      real(wp), allocatable :: first_raw_qat(:, :)
      real(wp), allocatable :: first_raw_dpat(:, :, :)
      real(wp), allocatable :: first_raw_qpat(:, :, :)
      real(wp), allocatable :: last_eelec(:)
   contains
      procedure :: before_solve => capture_before_solve
      procedure :: after_iteration => capture_after_iteration
      procedure :: finished => capture_finished
   end type recording_observer

contains

subroutine capture_before_solve(self, iteration, wfn, pot)
   class(recording_observer), intent(inout) :: self
   integer, intent(in) :: iteration
   type(wavefunction_type), intent(in) :: wfn
   type(potential_type), intent(in) :: pot

   self%before_count = self%before_count + 1
   self%sequence_ok = self%sequence_ok .and. iteration == self%before_count
   self%shapes_ok = self%shapes_ok &
      & .and. size(wfn%coeff, 1) == size(wfn%coeff, 2) &
      & .and. size(wfn%coeff, 3) == wfn%nspin &
      & .and. all(shape(wfn%qsh) == [size(pot%vsh, 1), size(pot%vsh, 2)]) &
      & .and. all(shape(wfn%dpat) == shape(pot%vdp)) &
      & .and. all(shape(wfn%qpat) == shape(pot%vqp))

   if (self%before_count == 1) then
      self%first_before_iteration = iteration
      self%first_hamiltonian = wfn%coeff
      self%first_mixed_qsh = wfn%qsh
      self%first_mixed_qat = wfn%qat
      self%first_mixed_dpat = wfn%dpat
      self%first_mixed_qpat = wfn%qpat
      self%first_vsh = pot%vsh
      self%first_vdp = pot%vdp
      self%first_vqp = pot%vqp
   end if
end subroutine capture_before_solve

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

   self%after_count = self%after_count + 1
   self%sequence_ok = self%sequence_ok .and. iteration == self%after_count
   self%last_after_iteration = iteration
   self%last_pnorm = pnorm
   self%last_elast = elast
   self%last_econverged = econverged
   self%last_pconverged = pconverged
   self%last_tconverged = tconverged
   self%last_converged = converged
   self%last_eelec = eelec

   if (self%after_count == 1) then
      self%first_after_iteration = iteration
      self%first_pnorm = pnorm
      self%first_coeff = wfn%coeff
      self%first_emo = wfn%emo
      self%first_focc = wfn%focc
      self%first_density = wfn%density
      self%first_raw_qsh = wfn%qsh
      self%first_raw_qat = wfn%qat
      self%first_raw_dpat = wfn%dpat
      self%first_raw_qpat = wfn%qpat
   end if

   self%shapes_ok = self%shapes_ok &
      & .and. size(wfn%coeff, 1) == size(wfn%coeff, 2) &
      & .and. size(wfn%coeff, 3) == wfn%nspin &
      & .and. size(wfn%emo, 2) == wfn%nspin &
      & .and. size(wfn%focc, 2) == max(2, wfn%nspin) &
      & .and. size(wfn%qpat, 1) == 6
   self%shapes_ok = self%shapes_ok &
      & .and. all(ieee_is_finite(wfn%coeff)) &
      & .and. all(ieee_is_finite(wfn%emo)) &
      & .and. all(ieee_is_finite(wfn%focc)) &
      & .and. all(ieee_is_finite(wfn%density))
end subroutine capture_after_iteration

subroutine capture_finished(self, iterations, status)
   class(recording_observer), intent(inout) :: self
   integer, intent(in) :: iterations
   integer, intent(in) :: status

   self%finished_count = self%finished_count + 1
   self%finished_iterations = iterations
   self%finished_status = status
end subroutine capture_finished

end module gpuxtb_tblite_observer_probe_type


program gpuxtb_tblite_observer_probe
   use mctc_env, only : wp, error_type
   use mctc_io, only : structure_type, new
   use tblite_context_type, only : context_type
   use tblite_results, only : results_type
   use tblite_scf, only : scf_observer, scf_observer_status_converged, &
      & scf_observer_status_failed, scf_observer_status_max_iterations
   use tblite_wavefunction, only : wavefunction_type, new_wavefunction
   use tblite_xtb_calculator, only : xtb_calculator
   use tblite_xtb_gfn2, only : new_gfn2_calculator
   use tblite_xtb_singlepoint, only : xtb_singlepoint
   use gpuxtb_tblite_observer_probe_type, only : recording_observer
   implicit none

   real(wp), parameter :: kt = 300.0_wp * 3.166808578545117e-6_wp
   real(wp), parameter :: accuracy = 1.0_wp
   real(wp), parameter :: xyz(3, 3) = reshape([ &
      & -0.47073898552969_wp,  0.81534384004086_wp, 0.0_wp, &
      & -0.47073898552969_wp, -0.81534384004086_wp, 0.0_wp, &
      &  0.94147797105939_wp,  0.0_wp,              0.0_wp], [3, 3])

   type(context_type) :: baseline_ctx, no_op_ctx, observed_ctx, limited_ctx, invalid_ctx
   type(structure_type) :: mol
   type(xtb_calculator) :: calc, limited_calc, invalid_calc
   type(wavefunction_type) :: baseline_wfn, no_op_wfn, observed_wfn, limited_wfn, &
      & invalid_wfn
   type(results_type) :: observed_results
   type(scf_observer) :: no_op
   type(recording_observer) :: observed, limited, invalid
   type(error_type), allocatable :: error
   real(wp) :: baseline_energy, no_op_energy, observed_energy, limited_energy, &
      & invalid_energy

   call new(mol, [1, 1, 1], xyz, charge=1.0_wp)
   call new_gfn2_calculator(calc, mol, error)
   if (allocated(error)) error stop "could not construct the GFN2 calculator"

   call new_wavefunction(baseline_wfn, mol%nat, calc%bas%nsh, calc%bas%nao, 1, kt)
   call xtb_singlepoint(baseline_ctx, mol, calc, baseline_wfn, accuracy, &
      & baseline_energy, verbosity=0)
   if (baseline_ctx%failed()) error stop "baseline H3+ calculation failed"

   call new_wavefunction(no_op_wfn, mol%nat, calc%bas%nsh, calc%bas%nao, 1, kt)
   call xtb_singlepoint(no_op_ctx, mol, calc, no_op_wfn, accuracy, no_op_energy, &
      & verbosity=0, observer=no_op)
   if (no_op_ctx%failed()) error stop "no-op-observer H3+ calculation failed"
   call assert_same_result(baseline_energy, baseline_wfn, no_op_energy, no_op_wfn)

   calc%save_integrals = .true.
   call new_wavefunction(observed_wfn, mol%nat, calc%bas%nsh, calc%bas%nao, 1, kt)
   call xtb_singlepoint(observed_ctx, mol, calc, observed_wfn, accuracy, &
      & observed_energy, verbosity=0, results=observed_results, observer=observed)
   if (observed_ctx%failed()) error stop "recording-observer H3+ calculation failed"
   call assert_same_result(baseline_energy, baseline_wfn, observed_energy, observed_wfn)
   call assert_observed_trace(observed, observed_results)

   limited_calc = calc
   limited_calc%save_integrals = .false.
   limited_calc%max_iter = 1
   call new_wavefunction(limited_wfn, mol%nat, limited_calc%bas%nsh, &
      & limited_calc%bas%nao, 1, kt)
   call xtb_singlepoint(limited_ctx, mol, limited_calc, limited_wfn, accuracy, &
      & limited_energy, verbosity=0, observer=limited)
   if (.not.limited_ctx%failed()) error stop "one-cycle calculation unexpectedly converged"
   if (limited%before_count /= 1 .or. limited%after_count /= 1) &
      & error stop "max-iteration callbacks were not paired"
   if (limited%finished_count /= 1 .or. limited%finished_iterations /= 1) &
      & error stop "max-iteration terminal callback is inconsistent"
   if (limited%finished_status /= scf_observer_status_max_iterations) &
      & error stop "max-iteration status was not observed"

   invalid_calc = calc
   invalid_calc%save_integrals = .false.
   invalid_calc%mixer_input%scf = 0
   call new_wavefunction(invalid_wfn, mol%nat, invalid_calc%bas%nsh, &
      & invalid_calc%bas%nao, 1, kt)
   call xtb_singlepoint(invalid_ctx, mol, invalid_calc, invalid_wfn, accuracy, &
      & invalid_energy, verbosity=0, observer=invalid)
   if (.not.invalid_ctx%failed()) error stop "invalid mixer unexpectedly succeeded"
   if (invalid%before_count /= 0 .or. invalid%after_count /= 0) &
      & error stop "invalid mixer emitted iteration callbacks"
   if (invalid%finished_count /= 1 .or. invalid%finished_iterations /= 0) &
      & error stop "invalid-mixer terminal callback is inconsistent"
   if (invalid%finished_status /= scf_observer_status_failed) &
      & error stop "invalid-mixer failure status was not observed"

   print '(a,1x,i0,1x,a)', "tblite SCC observer probe passed:", &
      & observed%after_count, "iterations"

contains

subroutine assert_same_result(reference_energy, reference, actual_energy, actual)
   real(wp), intent(in) :: reference_energy, actual_energy
   type(wavefunction_type), intent(in) :: reference, actual

   if (actual_energy /= reference_energy) error stop "observer changed the energy bits"
   if (actual%kt /= reference%kt) error stop "observer changed electronic temperature"
   if (any(actual%density /= reference%density)) error stop "observer changed density"
   if (any(actual%coeff /= reference%coeff)) error stop "observer changed coefficients"
   if (any(actual%emo /= reference%emo)) error stop "observer changed eigenvalues"
   if (any(actual%focc /= reference%focc)) error stop "observer changed occupations"
   if (any(actual%qsh /= reference%qsh)) error stop "observer changed shell charges"
   if (any(actual%qat /= reference%qat)) error stop "observer changed atomic charges"
   if (any(actual%dpat /= reference%dpat)) error stop "observer changed dipoles"
   if (any(actual%qpat /= reference%qpat)) error stop "observer changed quadrupoles"
end subroutine assert_same_result

subroutine assert_observed_trace(observer, results)
   type(recording_observer), intent(in) :: observer
   type(results_type), intent(in) :: results
   real(wp), allocatable :: lhs(:, :), rhs(:, :)
   real(wp) :: reconstructed_pnorm, eigen_tolerance, scale
   integer :: iorb, residual_size

   if (observer%before_count <= 1) error stop "probe did not exercise multiple SCC iterations"
   if (observer%before_count /= observer%after_count) &
      & error stop "pre/post callback counts differ"
   if (.not.observer%sequence_ok) error stop "callback iteration sequence is invalid"
   if (.not.observer%shapes_ok) error stop "callback shapes or finite values are invalid"
   if (observer%first_before_iteration /= 1 .or. observer%first_after_iteration /= 1) &
      & error stop "first callback does not describe iteration one"
   if (observer%last_after_iteration /= observer%after_count) &
      & error stop "last callback iteration is inconsistent"
   if (observer%finished_count /= 1) error stop "finished callback count is invalid"
   if (observer%finished_iterations /= observer%after_count) &
      & error stop "finished iteration count is inconsistent"
   if (observer%finished_status /= scf_observer_status_converged) &
      & error stop "converged status was not observed"
   if (.not.observer%last_converged .or. .not.observer%last_econverged &
      & .or. .not.observer%last_pconverged .or. .not.observer%last_tconverged) &
      & error stop "last convergence flags are inconsistent"

   residual_size = size(observer%first_mixed_qsh) &
      & + size(observer%first_mixed_dpat) + size(observer%first_mixed_qpat)
   reconstructed_pnorm = sqrt(( &
      & sum((observer%first_raw_qsh - observer%first_mixed_qsh)**2) &
      & + sum((observer%first_raw_dpat - observer%first_mixed_dpat)**2) &
      & + sum((observer%first_raw_qpat - observer%first_mixed_qpat)**2)) &
      & / real(residual_size, wp))
   if (abs(reconstructed_pnorm - observer%first_pnorm) &
      & > 1000.0_wp * epsilon(1.0_wp) * max(1.0_wp, observer%first_pnorm)) &
      & error stop "mixed/raw snapshots do not reconstruct the mixer RMS"

   allocate(lhs(size(observer%first_hamiltonian, 1), &
      & size(observer%first_hamiltonian, 2)))
   allocate(rhs, mold=lhs)
   do iorb = 1, size(lhs, 2)
      lhs(:, iorb) = matmul(observer%first_hamiltonian(:, :, 1), &
         & observer%first_coeff(:, iorb, 1))
      rhs(:, iorb) = observer%first_emo(iorb, 1) * matmul(results%overlap, &
         & observer%first_coeff(:, iorb, 1))
   end do
   scale = max(1.0_wp, maxval(abs(lhs)), maxval(abs(rhs)))
   eigen_tolerance = 1.0e-10_wp * scale
   if (maxval(abs(lhs - rhs)) > eigen_tolerance) &
      & error stop "before_solve did not capture the effective Hamiltonian"
   if (all(observer%first_hamiltonian == observer%first_coeff)) &
      & error stop "pre-solve Hamiltonian was already overwritten"
end subroutine assert_observed_trace

end program gpuxtb_tblite_observer_probe
