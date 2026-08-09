! Command-line driver for the xtbloom oracle trace recorder.
!
! Reads one case.spec file with this fixed layout (one token per line):
!   nat
!   atomic numbers (nat integers, one per line)
!   positions (3*nat reals, one per line, x1 y1 z1 x2 y2 z2 ...)
!   molecular_charge
!   unpaired_electrons
!   temperature (kelvin)
!   mixer_memory (integer)
!   mixer_damping
!   maximum_iterations (integer)
!   n_point_charges (integer; 0 disables PCEM)
!   optionally: per point charge, positions (3 reals), charge, hardness (5*n reals)
!
! Then runs the pinned GFN2 single point with the observer and streams the raw
! trace to stdout in the format parsed by generate_scc_corpus.py.
program scc_trace_main
   use mctc_env, only : wp
   use xtbloom_tblite_scc_trace_recorder, only : play
   implicit none

   integer :: nat, i, ip, npc, unpaired_electrons, mixer_memory, maximum_iterations
   integer, allocatable :: atomic_numbers(:)
   real(wp), allocatable :: positions(:, :)
   real(wp) :: molecular_charge, temperature, mixer_damping
   real(wp), allocatable :: pc_positions(:, :), pc_charges(:), pc_gammas(:)
   character(len=256) :: path

   if (command_argument_count() < 1) then
      write(*, '(a)') "usage: scc_trace_main <case.spec>"
      stop 64
   end if
   call get_command_argument(1, path)
   open(unit=10, file=trim(path), status='old', action='read')
   read(10, *) nat
   allocate(atomic_numbers(nat))
   do i = 1, nat
      read(10, *) atomic_numbers(i)
   end do
   allocate(positions(3, nat))
   do i = 1, nat
      read(10, *) positions(:, i)
   end do
   read(10, *) molecular_charge
   read(10, *) unpaired_electrons
   read(10, *) temperature
   read(10, *) mixer_memory
   read(10, *) mixer_damping
   read(10, *) maximum_iterations
   read(10, *) npc
   if (npc > 0) then
      allocate(pc_positions(3, npc), pc_charges(npc), pc_gammas(npc))
      do ip = 1, npc
         read(10, *) pc_positions(:, ip), pc_charges(ip), pc_gammas(ip)
      end do
   end if
   close(10)

   if (npc > 0) then
      call play(atomic_numbers, positions, molecular_charge, unpaired_electrons, &
         & temperature, mixer_memory, mixer_damping, maximum_iterations, &
         & pc_positions, pc_charges, pc_gammas)
   else
      call play(atomic_numbers, positions, molecular_charge, unpaired_electrons, &
         & temperature, mixer_memory, mixer_damping, maximum_iterations)
   end if
end program scc_trace_main
