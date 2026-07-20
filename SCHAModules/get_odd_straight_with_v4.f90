
! This subroutine computes the odd (third order) SCHA correction to the
! free energy Hessian, including the fourth order term v4:
!
!     phi_sc_odd = v3^T . Lambda . (I - v4.Lambda)^-1 . v3
!
! (R. Bianco et al., PRB 96, 014111 (2017), Eq. 27), with nl = n_mode^2.
!
! MEMORY-OPTIMIZED "KRON" VERSION: the nl x nl Lambda matrix is NEVER
! materialized.  The old path called get_cmat, which allocated TWO extra
! nl x nl temporaries (mat_e, mat_et) plus the nl x nl output cmat and did
! an O(N^6) dgemm; together with v4 that was a 4-array (4 "d4 units") peak.
! Instead we exploit the exact factorized structure that get_cmat builds
! (see get_cmat.f90, loops at its lines ~78-90):
!
!   mat_e (ka,ja) = e(nu,x) * e(mu,y)
!   mat_et(ja,ka) = mat_e(ka,ja) * mat_w(mu,nu) * 0.5
!   Lambda        = mat_e . mat_et
!
! with ka = (x-1)*N + y  (y fast)  and  ja = (mu-1)*N + nu  (nu fast), i.e.
!
!   Lambda(ka,ka') = sum_{mu,nu} e(nu,x) e(mu,y) D(mu,nu) e(nu,x') e(mu,y')
!
! where e(N,N) comes from get_emat, D(mu,nu) = mat_w(mu,nu)/2 with mat_w
! from get_g (both small, O(N^2)).  Applying Lambda therefore reduces to
! contractions with the SMALL matrix e (each an O(N^5) dgemm) plus a
! diagonal scaling over the mode pair (O(N^4)).  The only O(N^6) operation
! left is the inversion of (I - v4.Lambda).
!
! EXACT REPRODUCTION OF THE ORIGINAL PRODUCT (index bookkeeping).
! The original code built  maux = I - v42 . Lambda  with the reordered copy
!   v42(ja,ka) = v4(w,z,x,y),  ja = (w-1)*N + z (z fast),
!                              ka = (x-1)*N + y (y fast).
! Expanding, the matrix subtracted from the identity is, as a 4-tensor,
!
!   P(w,z,x,y) = sum_{x',y',mu,nu} v4(w,z,x',y') e(nu,x') e(mu,y')
!                                  * D(mu,nu) * e(nu,x) e(mu,y)
!
! placed at  maux( (w-1)*N + z , (x-1)*N + y ).  NO permutation symmetry of
! v4 is assumed anywhere: this is an identity in the indices, valid for
! arbitrary v4 (verified numerically against v1.5 with a NON-symmetrized
! random v4 as well; the old optional flag use_v4_symmetry is gone since
! the v42 copy it avoided no longer exists).
!
! P is evaluated with partial (Kronecker-factor) contractions that
! ping-pong between exactly TWO nl x nl buffers: maux and v4 ITSELF used
! as scratch.  Layouts below are column-major, leftmost index fastest:
!
!   A: B(w,z,x',mu)  = sum_y'  v4(w,z,x',y') e(mu,y')   v4  -> maux
!        one dgemm: (N^3 x N) . (N x N)^T, O(N^5)
!   B: T(w,z,nu,mu)  = sum_x'  B(w,z,x',mu) e(nu,x')    maux-> v4
!        N slice dgemms over mu: (N^2 x N) . (N x N)^T, O(N^5)
!   C: T(w,z,nu,mu) *= D(mu,nu)                          v4 in place, O(N^4)
!   D: C(w,z,nu,y)   = sum_mu  T(w,z,nu,mu) e(mu,y)     v4  -> maux
!        one dgemm: (N^3 x N) . (N x N), O(N^5)
!   E: P(w,z,x,y)    = sum_nu  C(w,z,nu,y) e(nu,x)      maux-> v4
!        N slice dgemms over y: (N^2 x N) . (N x N), O(N^5)
!   F: maux((w-1)N+z,(x-1)N+y) = delta - P(w,z,x,y)      v4  -> maux, O(N^4)
!        explicit permuted copy: the natural buffer layout of P has (w fast,
!        z slow) in the row pair and (x fast, y slow) in the column pair,
!        while the original v42.Lambda uses (z fast, w slow) rows and
!        (y fast, x slow) columns, so BOTH index pairs are swapped here.
!
! After F: LU inversion in place in maux (dgetrf/dgetri, unchanged), then
!   tmp = maux . v3^T                (nl x ns, O(N^5))
!   cf  = Lambda . tmp               via the same e/D/e^T factorization
!                                    applied to skinny matrices, using two
!                                    O(N^3) buffers (N x N x ns)
!   phi_sc_odd = v32 . cf            (as before)
!
! !!!  WARNING: v4 IS INTENT(INOUT) AND ITS CONTENT IS DESTROYED  !!!
! v4 is deliberately used as one of the two nl x nl scratch buffers (its
! original content is consumed by step A before step B overwrites it).
! This is safe for the only caller, Ensemble.get_free_energy_hessian
! (Modules/Ensemble.py): it creates d4 with SCHAModules.get_v4 (hence
! F-contiguous, so f2py's intent(inout) wraps it in place without a copy),
! optionally symmetrizes it in place, calls this routine, and never uses
! d4 again.  Any NEW caller must pass a throw-away, F-contiguous v4.
!
! Peak large-memory budget: exactly 2 nl x nl arrays alive (v4 + maux),
! plus O(N^3) skinny buffers (v32, tmp, cf, b1, b2) and O(N^2)/O(N) work
! arrays.  For N = 192 (Au 4x4x4): 2 * 10.87 GB = 21.7 GB, vs ~43.5 GB
! inside the old get_cmat call (v4 + mat_e + mat_et + cmat).

subroutine get_odd_straight_with_v4 ( a, wr, er, transmode, amass, ityp_sc, T, v3, v4, phi_sc_odd, &
  n_mode, nat_sc, ntyp)

  implicit none

  double precision, dimension(n_mode), intent(in) :: a, wr
  double precision, dimension(nat_sc,n_mode,3), intent(in) :: er
  logical, dimension(n_mode), intent(in) :: transmode
  double precision, dimension(ntyp), intent(in) :: amass
  integer, dimension(nat_sc), intent(in) :: ityp_sc
  double precision, intent(in) :: T
  double precision, dimension(n_mode,n_mode,n_mode), intent(in) :: v3
  ! v4 is used as scratch and DESTROYED on exit -- see warning above.
  double precision, dimension(n_mode,n_mode,n_mode,n_mode), intent(inout) :: v4
  double precision, dimension(n_mode, n_mode), intent(out) :: phi_sc_odd

  integer :: nat_sc, n_mode, nl, ns, ntyp

  ! The ONLY allocated nl x nl array (v4, the other big buffer, is the
  ! caller's own array).
  double precision, dimension(:,:), allocatable :: maux

  ! Small O(N^2) factors of Lambda.
  double precision, dimension(:,:), allocatable :: e     ! from get_emat
  double precision, dimension(:,:), allocatable :: dmat  ! D = mat_w/2, from get_g

  ! Skinny O(N^3) buffers.
  double precision, dimension(:,:), allocatable :: v32       ! ns x nl
  double precision, dimension(:,:), allocatable :: tmp, cf   ! nl x ns
  double precision, dimension(:,:,:), allocatable :: b1, b2  ! ns x ns x ns

  double precision, dimension(:), allocatable :: work
  integer, dimension(:), allocatable :: ipiv
  integer :: info

  integer :: mu, nu
  integer :: ka
  integer :: i, s, x, y, z, w

  logical, parameter :: debug = .true.

  if (debug) then
    print *, "=== DEBUG ODD STRAIGHT (kron, Lambda never materialized) ==="
    print *, "N_MODE:", n_mode
    print *, "NTYP:", ntyp
    print *, "NAT_SC:", nat_sc
    call flush()
  end if

  ns = n_mode
  nl = n_mode*n_mode

  allocate(maux(nl,nl))
  allocate(e(ns,ns))
  allocate(dmat(ns,ns))
  allocate(v32(ns,nl))
  allocate(tmp(nl,ns))
  allocate(cf(nl,ns))
  allocate(b1(ns,ns,ns))
  allocate(b2(ns,ns,ns))
  allocate(ipiv(nl))
  allocate(work(nl))

  ! Small factors of Lambda: exactly what get_cmat used internally.
  ! (v3_log = .true., as in the old get_cmat call from this routine.)
  call get_emat ( er, a, amass, ityp_sc, .true., transmode, e, n_mode, nat_sc, ntyp)
  call get_g (a, wr, transmode, T, dmat, n_mode)
  dmat = 0.5d0 * dmat     ! D(mu,nu) = mat_w(mu,nu)/2 (the 0.5 of mat_et)

  ! Third order force constants as rank 2, v32(:,ka) with ka=(x-1)*N+y.
  ka = 0
  do x = 1, ns
    do y = 1, ns
      ka = ka + 1
      v32(:,ka) = v3(:,x,y)
    end do
  end do

  ! ---------------------------------------------------------------------
  ! Build maux = I - v42.Lambda without forming Lambda (steps A-F above).
  ! ---------------------------------------------------------------------

  ! A: B(w,z,x',mu) = sum_y' v4(w,z,x',y') e(mu,y')        [v4 -> maux]
  !    v4 viewed as (N^3 x N) with columns y'; op(B)=e^T has entry
  !    (y',mu) = e(mu,y').  Fills maux completely (N^3 * N = nl*nl).
  call dgemm('N','T', nl*ns, ns, ns, 1.0d0, v4(1,1,1,1), nl*ns, &
             e(1,1), ns, 0.0d0, maux(1,1), nl*ns)

  ! B: T(w,z,nu,mu) = sum_x' B(w,z,x',mu) e(nu,x')         [maux -> v4]
  !    For each mu, the slice B(:,:,:,mu) is the (N^2 x N) block starting
  !    at maux(1,(mu-1)*ns+1) (linear offset (mu-1)*N^3), columns x';
  !    result panel written at v4(:,:,:,mu) with layout (w,z,nu).
  !    From here on the original content of v4 is destroyed.
  do mu = 1, ns
    call dgemm('N','T', nl, ns, ns, 1.0d0, maux(1,(mu-1)*ns+1), nl, &
               e(1,1), ns, 0.0d0, v4(1,1,1,mu), nl)
  end do

  ! C: T(w,z,nu,mu) *= D(mu,nu)                            [v4 in place]
  !    NOTE the argument order: 4th dim of T is mu, 3rd is nu.
  do mu = 1, ns
    do nu = 1, ns
      v4(:,:,nu,mu) = v4(:,:,nu,mu) * dmat(mu,nu)
    end do
  end do

  ! D: C(w,z,nu,y) = sum_mu T(w,z,nu,mu) e(mu,y)           [v4 -> maux]
  call dgemm('N','N', nl*ns, ns, ns, 1.0d0, v4(1,1,1,1), nl*ns, &
             e(1,1), ns, 0.0d0, maux(1,1), nl*ns)

  ! E: P(w,z,x,y) = sum_nu C(w,z,nu,y) e(nu,x)             [maux -> v4]
  do y = 1, ns
    call dgemm('N','N', nl, ns, ns, 1.0d0, maux(1,(y-1)*ns+1), nl, &
               e(1,1), ns, 0.0d0, v4(1,1,1,y), nl)
  end do

  ! F: maux = I - P with BOTH index pairs swapped to the original
  !    v42.Lambda convention: rows (z fast, w slow), cols (y fast, x slow).
  do y = 1, ns
    do x = 1, ns
      ka = (x-1)*ns + y
      do z = 1, ns
        do w = 1, ns
          maux((w-1)*ns + z, ka) = -v4(w,z,x,y)
        end do
      end do
    end do
  end do
  do i = 1, nl
    maux(i,i) = maux(i,i) + 1.0d0
  end do

  ! Invert ** iden - v4 lamat ** in place (unchanged from v1.5).
  call dgetrf ( nl, nl, maux, nl, ipiv, info )
  call dgetri ( nl, maux, nl, ipiv, work, nl, info )

  ! tmp = (I - v4 lamat)^-1 . v3^T                         (nl x ns)
  call dgemm('N','T', nl, ns, nl, 1.0d0, maux(1,1), nl, &
             v32(1,1), ns, 0.0d0, tmp(1,1), nl)

  ! ---------------------------------------------------------------------
  ! cf = Lambda . tmp, again without forming Lambda.  With tmp's row index
  ! ka' = (x'-1)*N + y' (y' fast) read as tmp3(y',x',s):
  !   u(mu,nu,s) = sum_{x',y'} e(mu,y') e(nu,x') tmp3(y',x',s)
  !   cf3(y,x,s) = sum_{mu,nu} e(mu,y) e(nu,x) D(mu,nu) u(mu,nu,s)
  ! All buffers are O(N^3).
  ! ---------------------------------------------------------------------

  ! C1: b1(mu,x',s) = sum_y' e(mu,y') tmp3(y',x',s)
  !     tmp reshaped as (N x N*ns), one dgemm.
  call dgemm('N','N', ns, nl, ns, 1.0d0, e(1,1), ns, &
             tmp(1,1), ns, 0.0d0, b1(1,1,1), ns)

  ! C2: b2(mu,nu,s) = sum_x' b1(mu,x',s) e(nu,x')
  do s = 1, ns
    call dgemm('N','T', ns, ns, ns, 1.0d0, b1(1,1,s), ns, &
               e(1,1), ns, 0.0d0, b2(1,1,s), ns)
  end do

  ! C3: b2(mu,nu,s) *= D(mu,nu)
  do s = 1, ns
    b2(:,:,s) = b2(:,:,s) * dmat(:,:)
  end do

  ! C4: b1(y,nu,s) = sum_mu e(mu,y) b2(mu,nu,s)   (e^T . b2_s)
  do s = 1, ns
    call dgemm('T','N', ns, ns, ns, 1.0d0, e(1,1), ns, &
               b2(1,1,s), ns, 0.0d0, b1(1,1,s), ns)
  end do

  ! C5: cf3(y,x,s) = sum_nu b1(y,nu,s) e(nu,x); column s of cf viewed as
  !     an (N x N) block (y fast, x slow), matching ka = (x-1)*N + y.
  do s = 1, ns
    call dgemm('N','N', ns, ns, ns, 1.0d0, b1(1,1,s), ns, &
               e(1,1), ns, 0.0d0, cf(1,s), ns)
  end do

  ! phi_sc_odd = v3 . lamat (I - v4 lamat)^-1 . v3
  call dgemm('N','N', ns, ns, nl, 1.0d0, v32(1,1), ns, &
             cf(1,1), nl, 0.0d0, phi_sc_odd(1,1), ns)

  deallocate(maux, e, dmat, v32, tmp, cf, b1, b2, ipiv, work)

end subroutine get_odd_straight_with_v4
