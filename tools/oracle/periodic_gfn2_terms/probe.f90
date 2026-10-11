! SPDX-License-Identifier: GPL-3.0-or-later
!
! Standalone term probe for the pinned tblite periodic GFN2 oracle.
!
! This program deliberately uses tblite's Fortran modules rather than the
! xTBloom implementation.  It emits a simple, lossless line protocol that the
! Python fixture tool normalizes into canonical JSON.  Array values are emitted
! in Fortran column-major order and every logical shape is explicit.
program periodic_gfn2_terms_probe
   use mctc_env, only : wp, error_type
   use mctc_io, only : structure_type, new
   use tblite_adjlist, only : adjacency_list, new_adjacency_list
   use tblite_basis_type, only : get_cutoff
   use tblite_container_cache, only : container_cache
   use tblite_coulomb_cache, only : coulomb_cache
   use tblite_cutoff, only : get_lattice_points
   use tblite_disp_cache, only : dispersion_cache
   use tblite_scf_potential, only : potential_type, new_potential
   use tblite_wavefunction_type, only : wavefunction_type, new_wavefunction
   use tblite_xtb_calculator, only : xtb_calculator
   use tblite_xtb_gfn2, only : new_gfn2_calculator
   use tblite_xtb_h0, only : get_hamiltonian, get_selfenergy
   implicit none

   character(len=64) :: mode
   character(len=1024) :: input_path
   type(structure_type) :: mol
   type(xtb_calculator) :: calc
   type(error_type), allocatable :: error
   integer, allocatable :: numbers(:)
   real(wp), allocatable :: qsh(:), qat(:), dpat(:, :), qpat(:, :)

   if (command_argument_count() /= 2) then
      error stop "usage: periodic-gfn2-terms-probe MODE INPUT"
   end if
   call get_command_argument(1, mode)
   call get_command_argument(2, input_path)

   call read_fixture(trim(input_path), mol, numbers, qsh, qat, dpat, qpat)
   call new_gfn2_calculator(calc, mol, error)
   if (allocated(error)) error stop "pinned tblite could not construct GFN2 calculator"

   write(*, '(a)') "SCHEMA xtbloom-tblite-periodic-gfn2-term-probe-v1"
   write(*, '(a,1x,a)') "MODE", trim(mode)
   call emit_integer_1("atomic_numbers", numbers)
   call emit_real_2("positions_bohr", mol%xyz)
   call emit_real_2("lattice_vectors_bohr_columns", mol%lattice)

   select case(trim(mode))
   case("cn")
      call probe_cn(mol, calc)
   case("repulsion")
      call probe_repulsion(mol, calc)
   case("d4")
      call probe_d4(mol, calc, qat)
   case("integrals-h0")
      call probe_integrals_h0(mol, calc)
   case("charge-ewald")
      call probe_charge_ewald(mol, calc, qsh, qat)
   case("multipoles")
      call probe_multipoles(mol, calc, qat, dpat, qpat)
   case default
      error stop "unknown probe mode"
   end select

contains

subroutine read_fixture(path, mol, numbers, qsh, qat, dpat, qpat)
   character(len=*), intent(in) :: path
   type(structure_type), intent(out) :: mol
   integer, allocatable, intent(out) :: numbers(:)
   real(wp), allocatable, intent(out) :: qsh(:), qat(:), dpat(:, :), qpat(:, :)

   integer :: unit, stat, nat, nsh, iat, ivec
   real(wp), allocatable :: xyz(:, :)
   real(wp) :: lattice(3, 3)

   open(newunit=unit, file=path, status="old", action="read", iostat=stat)
   if (stat /= 0) error stop "could not open fixture input"
   read(unit, *, iostat=stat) nat
   if (stat /= 0 .or. nat <= 0) error stop "invalid atom count"
   allocate(numbers(nat), xyz(3, nat), qat(nat), dpat(3, nat), qpat(6, nat))
   read(unit, *, iostat=stat) numbers
   if (stat /= 0) error stop "invalid atomic numbers"
   do iat = 1, nat
      read(unit, *, iostat=stat) xyz(:, iat)
      if (stat /= 0) error stop "invalid Cartesian positions"
   end do
   do ivec = 1, 3
      read(unit, *, iostat=stat) lattice(:, ivec)
      if (stat /= 0) error stop "invalid lattice vectors"
   end do
   read(unit, *, iostat=stat) nsh
   if (stat /= 0 .or. nsh <= 0) error stop "invalid shell-charge count"
   allocate(qsh(nsh))
   read(unit, *, iostat=stat) qsh
   if (stat /= 0) error stop "invalid shell charges"
   read(unit, *, iostat=stat) qat
   if (stat /= 0) error stop "invalid atomic charges"
   do iat = 1, nat
      read(unit, *, iostat=stat) dpat(:, iat)
      if (stat /= 0) error stop "invalid atomic dipoles"
   end do
   do iat = 1, nat
      read(unit, *, iostat=stat) qpat(:, iat)
      if (stat /= 0) error stop "invalid atomic quadrupoles"
   end do
   close(unit)

   ! mctc-lib stores the three direct lattice vectors as matrix columns.
   call new(mol, numbers, xyz, lattice=lattice)
end subroutine read_fixture


subroutine probe_cn(mol, calc)
   type(structure_type), intent(in) :: mol
   type(xtb_calculator), intent(in) :: calc

   real(wp), allocatable :: cn(:), dcndr(:, :, :), dcndL(:, :, :)

   allocate(cn(mol%nat), dcndr(3, mol%nat, mol%nat), dcndL(3, 3, mol%nat))
   call calc%ncoord%get_cn(mol, cn, dcndr, dcndL)
   call emit_real_1("gfn2_dexp_cn", cn)
   call emit_real_3("gfn2_dexp_dcndr", dcndr)
   call emit_real_3("gfn2_dexp_dcndL", dcndL)
end subroutine probe_cn


subroutine probe_repulsion(mol, calc)
   type(structure_type), intent(in) :: mol
   type(xtb_calculator), intent(in) :: calc

   type(container_cache) :: cache
   real(wp), allocatable :: energies(:), gradient(:, :)
   real(wp) :: sigma(3, 3)

   allocate(energies(mol%nat), gradient(3, mol%nat))
   energies = 0.0_wp
   gradient = 0.0_wp
   sigma = 0.0_wp
   call calc%repulsion%get_engrad(mol, cache, energies, gradient, sigma)
   call emit_real_1("per_atom_energy_hartree", energies)
   call emit_real_0("total_energy_hartree", sum(energies))
   call emit_real_2("gradient_hartree_per_bohr", gradient)
   call emit_real_2("strain_derivatives_hartree", sigma)
end subroutine probe_repulsion


subroutine probe_d4(mol, calc, qat)
   type(structure_type), intent(in) :: mol
   type(xtb_calculator), intent(in) :: calc
   real(wp), intent(in) :: qat(:)

   type(container_cache), target :: cache
   type(dispersion_cache), pointer :: dcache
   type(wavefunction_type) :: wfn
   type(potential_type) :: pot
   real(wp), allocatable :: nonsc_energy(:), pair_energy(:)
   real(wp), allocatable :: nonsc_gradient(:, :), pair_gradient(:, :)
   real(wp) :: nonsc_sigma(3, 3), pair_sigma(3, 3)

   call new_wavefunction(wfn, mol%nat, calc%bas%nsh, calc%bas%nao, 1, 0.0_wp)
   call new_potential(pot, mol, calc%bas, 1)
   wfn%qat(:, 1) = qat
   allocate(nonsc_energy(mol%nat), pair_energy(mol%nat))
   allocate(nonsc_gradient(3, mol%nat), pair_gradient(3, mol%nat))
   nonsc_energy = 0.0_wp
   pair_energy = 0.0_wp
   nonsc_gradient = 0.0_wp
   pair_gradient = 0.0_wp
   nonsc_sigma = 0.0_wp
   pair_sigma = 0.0_wp

   call calc%dispersion%update(mol, cache)
   call calc%dispersion%get_engrad(mol, cache, nonsc_energy, nonsc_gradient, nonsc_sigma)
   call calc%dispersion%get_energy(mol, cache, wfn, pair_energy)
   call pot%reset
   call calc%dispersion%get_potential(mol, cache, wfn, pot)
   call calc%dispersion%get_gradient(mol, cache, wfn, pair_gradient, pair_sigma)
   call view_dispersion_cache(cache, dcache)

   call emit_real_1("fixed_atomic_charges_e", qat)
   call emit_real_1("nonsc_per_atom_energy_hartree", nonsc_energy)
   call emit_real_1("pair_per_atom_energy_hartree", pair_energy)
   call emit_real_0("total_energy_hartree", sum(nonsc_energy) + sum(pair_energy))
   call emit_real_2("nonsc_gradient_hartree_per_bohr", nonsc_gradient)
   call emit_real_2("pair_gradient_hartree_per_bohr", pair_gradient)
   call emit_real_2("total_gradient_hartree_per_bohr", nonsc_gradient + pair_gradient)
   call emit_real_2("nonsc_strain_derivatives_hartree", nonsc_sigma)
   call emit_real_2("pair_strain_derivatives_hartree", pair_sigma)
   call emit_real_2("total_strain_derivatives_hartree", nonsc_sigma + pair_sigma)
   call emit_real_2("charge_potential_hartree_per_e", pot%vat)
   call emit_real_1("d4_cn", dcache%cn)
   call emit_real_3("d4_dcndr", dcache%dcndr)
   call emit_real_3("d4_dcndL", dcache%dcndL)
   call emit_real_4("d4_dispersion_matrix", dcache%dispmat)
   call emit_real_3("d4_weight_vector", dcache%gwvec)
   call emit_real_3("d4_weight_charge_derivative", dcache%dgwdq)
end subroutine probe_d4


subroutine probe_integrals_h0(mol, calc)
   type(structure_type), intent(in) :: mol
   type(xtb_calculator), intent(in) :: calc

   type(adjacency_list) :: list
   real(wp), allocatable :: cn(:), selfenergy(:), lattr(:, :)
   real(wp), allocatable :: overlap(:, :), dipole(:, :, :), quadrupole(:, :, :)
   real(wp), allocatable :: hamiltonian(:, :)
   real(wp) :: cutoff
   integer, allocatable :: shell_l(:)
   integer :: iat, isp, ish, offset

   allocate(cn(mol%nat), selfenergy(calc%bas%nsh), shell_l(calc%bas%nsh))
   call calc%ncoord%get_cn(mol, cn)
   call get_selfenergy(calc%h0, mol%id, calc%bas%ish_at, calc%bas%nsh_id, &
      & cn=cn, selfenergy=selfenergy)

   cutoff = get_cutoff(calc%bas, 1.0_wp)
   call get_lattice_points(mol%periodic, mol%lattice, cutoff, lattr)
   call new_adjacency_list(list, mol, lattr, cutoff)
   allocate(overlap(calc%bas%nao, calc%bas%nao), &
      & dipole(3, calc%bas%nao, calc%bas%nao), &
      & quadrupole(6, calc%bas%nao, calc%bas%nao), &
      & hamiltonian(calc%bas%nao, calc%bas%nao))
   call get_hamiltonian(mol, lattr, list, calc%bas, calc%h0, selfenergy, &
      & overlap, dipole, quadrupole, hamiltonian)

   offset = 0
   do iat = 1, mol%nat
      isp = mol%id(iat)
      do ish = 1, calc%bas%nsh_at(iat)
         shell_l(offset + ish) = calc%bas%cgto(ish, isp)%ang
      end do
      offset = offset + calc%bas%nsh_at(iat)
   end do

   call emit_integer_0("number_of_shells", calc%bas%nsh)
   call emit_integer_0("number_of_aos", calc%bas%nao)
   call emit_integer_1("shells_per_atom", calc%bas%nsh_at)
   call emit_integer_1("aos_per_shell", calc%bas%nao_sh)
   call emit_integer_1("ao_offsets_per_shell_zero_based", calc%bas%iao_sh)
   call emit_integer_1("shell_offsets_per_atom_zero_based", calc%bas%ish_at)
   call emit_integer_1("ao_to_atom_one_based", calc%bas%ao2at)
   call emit_integer_1("ao_to_shell_one_based", calc%bas%ao2sh)
   call emit_integer_1("shell_to_atom_one_based", calc%bas%sh2at)
   call emit_integer_1("shell_angular_momentum", shell_l)
   call emit_real_0("integral_cutoff_bohr", cutoff)
   call emit_real_2("integral_lattice_translations_bohr", lattr)
   call emit_real_1("gfn2_dexp_cn", cn)
   call emit_real_1("shell_self_energies_hartree", selfenergy)
   call emit_real_2("overlap_matrix", overlap)
   call emit_real_3("dipole_matrices_bohr", dipole)
   call emit_real_3("quadrupole_matrices_bohr2", quadrupole)
   call emit_real_2("h0_matrix_hartree", hamiltonian)
end subroutine probe_integrals_h0


subroutine probe_charge_ewald(mol, calc, qsh, qat)
   type(structure_type), intent(in) :: mol
   type(xtb_calculator), intent(in) :: calc
   real(wp), intent(in) :: qsh(:), qat(:)

   type(container_cache), target :: cache
   type(coulomb_cache), pointer :: ccache
   type(wavefunction_type) :: wfn
   type(potential_type) :: pot
   real(wp), allocatable :: energies(:), gradient(:, :)
   real(wp) :: sigma(3, 3)

   if (size(qsh) /= calc%bas%nsh) error stop "fixture shell charges do not match GFN2 basis"
   call new_wavefunction(wfn, mol%nat, calc%bas%nsh, calc%bas%nao, 1, 0.0_wp)
   call new_potential(pot, mol, calc%bas, 1)
   wfn%qsh(:, 1) = qsh
   wfn%qat(:, 1) = qat
   allocate(energies(mol%nat), gradient(3, mol%nat))
   energies = 0.0_wp
   gradient = 0.0_wp
   sigma = 0.0_wp

   call calc%coulomb%es2%update(mol, cache)
   call calc%coulomb%es2%get_energy(mol, cache, wfn, energies)
   call pot%reset
   call calc%coulomb%es2%get_potential(mol, cache, wfn, pot)
   call calc%coulomb%es2%get_gradient(mol, cache, wfn, gradient, sigma)
   call view_coulomb_cache(cache, ccache)

   call emit_integer_1("shells_per_atom", calc%coulomb%es2%nshell)
   call emit_integer_1("shell_offsets_per_atom_zero_based", calc%coulomb%es2%offset)
   call emit_real_1("fixed_shell_charges_e", qsh)
   call emit_real_1("fixed_atomic_charges_e", qat)
   call emit_real_0("ewald_alpha_bohr_inverse", ccache%alpha)
   call emit_real_2("shell_coulomb_matrix_hartree_per_e2", ccache%amat)
   call emit_real_1("per_atom_energy_hartree", energies)
   call emit_real_0("total_energy_hartree", sum(energies))
   call emit_real_2("shell_charge_potential_hartree_per_e", pot%vsh)
   call emit_real_2("gradient_hartree_per_bohr", gradient)
   call emit_real_2("strain_derivatives_hartree", sigma)
end subroutine probe_charge_ewald


subroutine probe_multipoles(mol, calc, qat, dpat, qpat)
   type(structure_type), intent(in) :: mol
   type(xtb_calculator), intent(in) :: calc
   real(wp), intent(in) :: qat(:), dpat(:, :), qpat(:, :)

   type(container_cache), target :: cache
   type(coulomb_cache), pointer :: ccache
   real(wp), allocatable :: zero_charge(:), zero_dipole(:, :), zero_quadrupole(:, :)

   allocate(zero_charge(mol%nat), zero_dipole(3, mol%nat), &
      & zero_quadrupole(6, mol%nat))
   zero_charge = 0.0_wp
   zero_dipole = 0.0_wp
   zero_quadrupole = 0.0_wp
   call calc%coulomb%aes2%update(mol, cache)
   call view_coulomb_cache(cache, ccache)

   call emit_real_0("ewald_alpha_multipole_bohr_inverse", ccache%alpha_multipole)
   call emit_real_1("multipole_cn", ccache%cn)
   call emit_real_1("multipole_damping_radii_bohr", ccache%mrad)
   call emit_real_1("damping_radius_cn_derivative_bohr", ccache%dmrdcn)
   call emit_real_3("charge_dipole_matrix", ccache%amat_sd)
   call emit_real_4("dipole_dipole_matrix", ccache%amat_dd)
   call emit_real_3("charge_quadrupole_matrix", ccache%amat_sq)

   call emit_multipole_state("dipole_only", mol, calc, cache, zero_charge, dpat, &
      & zero_quadrupole)
   call emit_multipole_state("charge_quadrupole", mol, calc, cache, qat, zero_dipole, qpat)
   call emit_multipole_state("full", mol, calc, cache, qat, dpat, qpat)
end subroutine probe_multipoles


subroutine emit_multipole_state(label, mol, calc, cache, qat, dpat, qpat)
   character(len=*), intent(in) :: label
   type(structure_type), intent(in) :: mol
   type(xtb_calculator), intent(in) :: calc
   type(container_cache), intent(inout) :: cache
   real(wp), intent(in) :: qat(:), dpat(:, :), qpat(:, :)

   type(wavefunction_type) :: wfn
   type(potential_type) :: pot
   real(wp), allocatable :: total(:), aes(:), axc(:), gradient(:, :)
   real(wp) :: sigma(3, 3)

   call new_wavefunction(wfn, mol%nat, calc%bas%nsh, calc%bas%nao, 1, 0.0_wp)
   call new_potential(pot, mol, calc%bas, 1)
   wfn%qat(:, 1) = qat
   wfn%dpat(:, :, 1) = dpat
   wfn%qpat(:, :, 1) = qpat
   allocate(total(mol%nat), aes(mol%nat), axc(mol%nat), gradient(3, mol%nat))
   total = 0.0_wp
   aes = 0.0_wp
   axc = 0.0_wp
   gradient = 0.0_wp
   sigma = 0.0_wp

   call calc%coulomb%aes2%get_energy(mol, cache, wfn, total)
   call calc%coulomb%aes2%get_energy_aes(mol, cache, wfn, aes)
   call calc%coulomb%aes2%get_energy_axc(mol, wfn, axc)
   call pot%reset
   call calc%coulomb%aes2%get_potential(mol, cache, wfn, pot)
   call calc%coulomb%aes2%get_gradient(mol, cache, wfn, gradient, sigma)

   call emit_real_1(label // "_fixed_atomic_charges_e", qat)
   call emit_real_2(label // "_fixed_atomic_dipoles_e_bohr", dpat)
   call emit_real_2(label // "_fixed_atomic_quadrupoles_e_bohr2", qpat)
   call emit_real_1(label // "_per_atom_total_energy_hartree", total)
   call emit_real_1(label // "_per_atom_aes_energy_hartree", aes)
   call emit_real_1(label // "_per_atom_axc_energy_hartree", axc)
   call emit_real_0(label // "_total_energy_hartree", sum(total))
   call emit_real_2(label // "_charge_potential_hartree_per_e", pot%vat)
   call emit_real_3(label // "_dipole_potential_hartree_per_e_bohr", pot%vdp)
   call emit_real_3(label // "_quadrupole_potential_hartree_per_e_bohr2", pot%vqp)
   call emit_real_2(label // "_gradient_hartree_per_bohr", gradient)
   call emit_real_2(label // "_strain_derivatives_hartree", sigma)
end subroutine emit_multipole_state


subroutine view_dispersion_cache(cache, ptr)
   type(container_cache), target, intent(inout) :: cache
   type(dispersion_cache), pointer, intent(out) :: ptr

   nullify(ptr)
   select type(raw => cache%raw)
   type is(dispersion_cache)
      ptr => raw
   end select
   if (.not.associated(ptr)) error stop "unexpected dispersion cache type"
end subroutine view_dispersion_cache


subroutine view_coulomb_cache(cache, ptr)
   type(container_cache), target, intent(inout) :: cache
   type(coulomb_cache), pointer, intent(out) :: ptr

   nullify(ptr)
   select type(raw => cache%raw)
   type is(coulomb_cache)
      ptr => raw
   end select
   if (.not.associated(ptr)) error stop "unexpected Coulomb cache type"
end subroutine view_coulomb_cache


subroutine emit_real_0(name, value)
   character(len=*), intent(in) :: name
   real(wp), intent(in) :: value
   write(*, '(a,1x,a,1x,i0)') "REAL", trim(name), 0
   write(*, '(es26.17e3)') value
end subroutine emit_real_0


subroutine emit_real_1(name, values)
   character(len=*), intent(in) :: name
   real(wp), intent(in) :: values(:)
   integer :: i
   write(*, '(a,1x,a,1x,i0,1x,i0)') "REAL", trim(name), 1, size(values, 1)
   do i = 1, size(values)
      write(*, '(es26.17e3)') values(i)
   end do
end subroutine emit_real_1


subroutine emit_real_2(name, values)
   character(len=*), intent(in) :: name
   real(wp), intent(in) :: values(:, :)
   integer :: i, j
   write(*, '(a,1x,a,1x,i0,2(1x,i0))') "REAL", trim(name), 2, shape(values)
   do j = 1, size(values, 2)
      do i = 1, size(values, 1)
         write(*, '(es26.17e3)') values(i, j)
      end do
   end do
end subroutine emit_real_2


subroutine emit_real_3(name, values)
   character(len=*), intent(in) :: name
   real(wp), intent(in) :: values(:, :, :)
   integer :: i, j, k
   write(*, '(a,1x,a,1x,i0,3(1x,i0))') "REAL", trim(name), 3, shape(values)
   do k = 1, size(values, 3)
      do j = 1, size(values, 2)
         do i = 1, size(values, 1)
            write(*, '(es26.17e3)') values(i, j, k)
         end do
      end do
   end do
end subroutine emit_real_3


subroutine emit_real_4(name, values)
   character(len=*), intent(in) :: name
   real(wp), intent(in) :: values(:, :, :, :)
   integer :: i, j, k, l
   write(*, '(a,1x,a,1x,i0,4(1x,i0))') "REAL", trim(name), 4, shape(values)
   do l = 1, size(values, 4)
      do k = 1, size(values, 3)
         do j = 1, size(values, 2)
            do i = 1, size(values, 1)
               write(*, '(es26.17e3)') values(i, j, k, l)
            end do
         end do
      end do
   end do
end subroutine emit_real_4


subroutine emit_integer_0(name, value)
   character(len=*), intent(in) :: name
   integer, intent(in) :: value
   write(*, '(a,1x,a,1x,i0)') "INTEGER", trim(name), 0
   write(*, '(i0)') value
end subroutine emit_integer_0


subroutine emit_integer_1(name, values)
   character(len=*), intent(in) :: name
   integer, intent(in) :: values(:)
   integer :: i
   write(*, '(a,1x,a,1x,i0,1x,i0)') "INTEGER", trim(name), 1, size(values, 1)
   do i = 1, size(values)
      write(*, '(i0)') values(i)
   end do
end subroutine emit_integer_1

end program periodic_gfn2_terms_probe
