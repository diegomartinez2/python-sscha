
! This subroutine calculates the L mat needed to get the average of the
! third order derivatives. It is formed by four polarization vectors
! times the mass^1/2 divided by the normal length.

subroutine get_odd_straight_with_v4 ( a, wr, er, transmode, amass, ityp_sc, T, v3, v4, phi_sc_odd, &
  n_mode, nat_sc, ntyp, use_v4_symmetry)

  implicit none

  double precision, dimension(n_mode), intent(in) :: a, wr
  double precision, dimension(nat_sc,n_mode,3), intent(in) :: er
  logical, dimension(n_mode), intent(in) :: transmode
  double precision, dimension(ntyp), intent(in) :: amass
  integer, dimension(nat_sc), intent(in) :: ityp_sc
  double precision, intent(in) :: T
  double precision, dimension(n_mode,n_mode,n_mode), intent(in) :: v3
  double precision, dimension(n_mode,n_mode,n_mode, n_mode), intent(in) :: v4
  double precision, dimension(n_mode, n_mode), intent(out) :: phi_sc_odd
  ! Optional: if present and .true., assume v4 is permutation-symmetric
  ! (standard case, after ApplySymmetryToTensor4 / use_symmetries=True) and
  ! skip building the explicit reordered copy v42 -- see note below.
  logical, intent(in), optional :: use_v4_symmetry


  integer :: nat_sc, n_mode, nl, ns, ntyp
  double precision, dimension(:,:), allocatable :: l, g, phi_aux, v1, v2, v32
  double precision :: lsum
  double precision, dimension(:), allocatable :: laux1, lres1, veclong
  double precision, dimension(:), allocatable :: laux2, lres2

  double precision, dimension(:,:), allocatable :: lamat, v42, maux
  double precision, dimension(:), allocatable :: work
  integer, dimension(:), allocatable :: ipiv
  integer :: info

  double precision, dimension(:,:), allocatable :: cf
  double precision, dimension(:,:), allocatable :: tmp

  integer :: mu, nu, alpha
  integer :: ka, ja
  integer :: i, j, x, y, z, w

  real :: t1, t2

  logical, parameter :: debug = .true.
  logical :: sym

  ! Get integers

  if (debug) then
    print *, "=== DEBUG ODD STRAIGHT ==="
    print *, "N_MODE:", n_mode
    print *, "NTYP:", ntyp
    print *, "NAT_SC:", nat_sc
    call flush()
  end if

  !nat_sc = size(er(:,1,1))
  !n_mode = 3*nat_sc

  ns = n_mode
  nl = n_mode*n_mode

  sym = .false.
  if (present(use_v4_symmetry)) sym = use_v4_symmetry

  ! Allocate stuff

  allocate(lamat(nl,nl))
  allocate(maux(nl,nl))
  allocate(ipiv(nl))
  allocate(work(nl))
  allocate(v32(n_mode,n_mode*n_mode))

  allocate(cf(nl,ns))
  allocate(tmp(nl,ns))

  ! v42 (explicit reordered copy of v4) is only needed when we cannot rely on
  ! the permutation symmetry of v4. It is one full nl x nl array.
  if (.not. sym) allocate(v42(nl,nl))

  ! Get lambda matrix

  call get_cmat ( a, wr, er, transmode, amass, ityp_sc, T, .true., lamat,n_mode, nat_sc, ntyp )

  !print *, "AFTER CMAT"
  !call flush()

  ! Write third and fourth order force constants as rank 2

  ka = 0

  do x = 1, n_mode
    do y = 1, n_mode
      ka = ka + 1
      v32(:,ka) = v3(:,x,y)
      if (.not. sym) then
        ja = 0
        do w = 1, n_mode
          do z = 1, n_mode
            ja = ja + 1
            v42(ja,ka) = v4(w,z,x,y)
          end do
        end do
      end if
    end do
  end do

  ! Prepare identity matrix directly in maux (was: iden then maux = iden;
  ! bitwise identical, avoids one nl x nl temporary)

  maux = 0.0d0

  do x = 1, nl
    maux(x,x) = 1.0d0
  end do

  ! Calculate ** iden - v4 lamat ** matrix

  !print *, "BEFORE I - V4Lambda"
  !call flush()

  if (.not. sym) then
    call dgemm('N','N',nl,nl,nl,-1.0d0,v42,nl,lamat,nl,1.0d0,maux,nl)
  else
    ! v4 fully permutation-symmetric => the reordered copy v42(ja,ka)=v4(w,z,x,y)
    ! equals the plain column-major reshape of v4 into an (nl,nl) matrix, so we
    ! can feed v4 directly to dgemm with leading dimension nl and skip v42.
    ! (Not valid with use_symmetries=False -> guarded by use_v4_symmetry.)
    call dgemm('N','N',nl,nl,nl,-1.0d0,v4,nl,lamat,nl,1.0d0,maux,nl)
  end if

  if (allocated(v42)) deallocate(v42)

  ! Invert ** iden - lamat v4 **

  !print *, "BEFORE (I - V4Lambda)^-1"
  !call flush()


  call dgetrf ( nl, nl, maux, nl, ipiv, info )
  call dgetri ( nl, maux, nl, ipiv, work, nl, info )

  ! Take product between lamat and the inverted matrix, contracted with v3.
  !
  ! REASSOCIATION (numerically equivalent to ~1e-15 relative, NOT bitwise):
  ! the original code formed the nl x nl product v42 := lamat * maux and then
  ! cf := v42 * v3^T. We instead evaluate  cf = lamat * (maux * v3^T) using an
  ! nl x ns buffer tmp, which is mathematically identical by associativity
  ! ( Lambda * M^-1 * v3^T = Lambda * (M^-1 * v3^T) ) and reduces both the flop
  ! count (O(N^6) -> O(N^5)) and the memory (no nl x nl buffer needed).

  ! tmp = (I - v4 lamat)^-1 * v3^T          (nl x ns)
  call dgemm('N','T',nl,ns,nl,1.0d0,maux,nl,&
             v32,ns,0.0d0,tmp,nl)

  ! cf = lamat * tmp = lamat (I - v4 lamat)^-1 v3^T   (nl x ns)
  call dgemm('N','N',nl,ns,nl,1.0d0,lamat,nl,&
             tmp,nl,0.0d0,cf,nl)

  ! Now get:
  ! v3 * ( 1 - lamat*v4)^-1 lamat *  v3

  !print *, "BEFORE V3 Lambda(I - V4Lambda)^-1 V3"
  !call flush()
  call dgemm('N','N',ns,ns,nl,1.0d0,v32,ns,&
             cf,nl,0.0d0,phi_sc_odd,ns)


  !call get_odd_from_cmat_fu2 (v42, v32, phi_sc_odd)

  ! Deallocate stuff

  deallocate(lamat,v32,maux,ipiv,work, cf, tmp)

end subroutine get_odd_straight_with_v4
