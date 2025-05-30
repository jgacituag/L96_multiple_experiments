



      subroutine dspdi(ap,n,kpvt,det,inert,work,job)
      integer n,job
      double precision ap(1),work(1)
      double precision det(2)
      integer kpvt(1),inert(3)
! 
! dspdi computes the determinant, inertia and inverse
! of a double precision symmetric matrix using the factors from
! dspfa, where the matrix is stored in packed form.
! 
! on entry
! 
! ap      double precision (n*(n+1)/2)
! the output from dspfa.
! 
! n       integer
! the order of the matrix a.
! 
! kpvt    integer(n)
! the pivot vector from dspfa.
! 
! work    double precision(n)
! work vector.  contents ignored.
! 
! job     integer
! job has the decimal expansion  abc  where
! if  c .ne. 0, the inverse is computed,
! if  b .ne. 0, the determinant is computed,
! if  a .ne. 0, the inertia is computed.
! 
! for example, job = 111  gives all three.
! 
! on return
! 
! variables not requested by job are not used.
! 
! ap     contains the upper triangle of the inverse of
! the original matrix, stored in packed form.
! the columns of the upper triangle are stored
! sequentially in a one-dimensional array.
! 
! det    double precision(2)
! determinant of original matrix.
! determinant = det(1) * 10.0**det(2)
! with 1.0 .le. dabs(det(1)) .lt. 10.0
! or det(1) = 0.0.
! 
! inert  integer(3)
! the inertia of the original matrix.
! inert(1)  =  number of positive eigenvalues.
! inert(2)  =  number of negative eigenvalues.
! inert(3)  =  number of zero eigenvalues.
! 
! error condition
! 
! a division by zero will occur if the inverse is requested
! and  dspco  has set rcond .eq. 0.0
! or  dspfa  has set  info .ne. 0 .
! 
! linpack. this version dated 08/14/78 .
! james bunch, univ. calif. san diego, argonne nat. lab.
! 
! subroutines and functions
! 
! blas daxpy,dcopy,ddot,dswap
! fortran dabs,iabs,mod
! 
! internal variables.
! 
      double precision akkp1,ddot,temp
      double precision ten,d,t,ak,akp1
      integer ij,ik,ikp1,iks,j,jb,jk,jkp1
      integer k,kk,kkp1,km1,ks,ksj,kskp1,kstep
      logical noinv,nodet,noert
! 
      noinv = mod(job,10) .eq. 0
      nodet = mod(job,100)/10 .eq. 0
      noert = mod(job,1000)/100 .eq. 0
! 
      if (nodet .and. noert) go to 140
         if (noert) go to 10
            inert(1) = 0
            inert(2) = 0
            inert(3) = 0
   10    continue
         if (nodet) go to 20
            det(1) = 1.0d0
            det(2) = 0.0d0
            ten = 10.0d0
   20    continue
         t = 0.0d0
         ik = 0
         do 130 k = 1, n
            kk = ik + k
            d = ap(kk)
! 
! check if 1 by 1
! 
            if (kpvt(k) .gt. 0) go to 50
! 
! 2 by 2 block
! use det (d  s)  =  (d/t * c - t) * t  ,  t = dabs(s)
! (s  c)
! to avoid underflow/overflow troubles.
! take two passes through scaling.  use  t  for flag.
! 
               if (t .ne. 0.0d0) go to 30
                  ikp1 = ik + k
                  kkp1 = ikp1 + k
                  t = dabs(ap(kkp1))
                  d = (d/t)*ap(kkp1+1) - t
               go to 40
   30          continue
                  d = t
                  t = 0.0d0
   40          continue
   50       continue
! 
            if (noert) go to 60
               if (d .gt. 0.0d0) inert(1) = inert(1) + 1
               if (d .lt. 0.0d0) inert(2) = inert(2) + 1
               if (d .eq. 0.0d0) inert(3) = inert(3) + 1
   60       continue
! 
            if (nodet) go to 120
               det(1) = d*det(1)
               if (det(1) .eq. 0.0d0) go to 110
   70             if (dabs(det(1)) .ge. 1.0d0) go to 80
                     det(1) = ten*det(1)
                     det(2) = det(2) - 1.0d0
                  go to 70
   80             continue
   90             if (dabs(det(1)) .lt. ten) go to 100
                     det(1) = det(1)/ten
                     det(2) = det(2) + 1.0d0
                  go to 90
  100             continue
  110          continue
  120       continue
            ik = ik + k
  130    continue
  140 continue
! 
! compute inverse(a)
! 
      if (noinv) go to 270
         k = 1
         ik = 0
  150    if (k .gt. n) go to 260
            km1 = k - 1
            kk = ik + k
            ikp1 = ik + k
            kkp1 = ikp1 + k
            if (kpvt(k) .lt. 0) go to 180
! 
! 1 by 1
! 
               ap(kk) = 1.0d0/ap(kk)
               if (km1 .lt. 1) go to 170
                  call dcopy(km1,ap(ik+1),1,work,1)
                  ij = 0
                  do 160 j = 1, km1
                     jk = ik + j
                     ap(jk) = ddot(j,ap(ij+1),1,work,1)
                     call daxpy(j-1,work(j),ap(ij+1),1,ap(ik+1),1)
                     ij = ij + j
  160             continue
                  ap(kk) = ap(kk) + ddot(km1,work,1,ap(ik+1),1)
  170          continue
               kstep = 1
            go to 220
  180       continue
! 
! 2 by 2
! 
               t = dabs(ap(kkp1))
               ak = ap(kk)/t
               akp1 = ap(kkp1+1)/t
               akkp1 = ap(kkp1)/t
               d = t*(ak*akp1 - 1.0d0)
               ap(kk) = akp1/d
               ap(kkp1+1) = ak/d
               ap(kkp1) = -akkp1/d
               if (km1 .lt. 1) go to 210
                  call dcopy(km1,ap(ikp1+1),1,work,1)
                  ij = 0
                  do 190 j = 1, km1
                     jkp1 = ikp1 + j
                     ap(jkp1) = ddot(j,ap(ij+1),1,work,1)
                     call daxpy(j-1,work(j),ap(ij+1),1,ap(ikp1+1),1)
                     ij = ij + j
  190             continue
                  ap(kkp1+1) = ap(kkp1+1)
     *                         + ddot(km1,work,1,ap(ikp1+1),1)
                  ap(kkp1) = ap(kkp1)
     *                       + ddot(km1,ap(ik+1),1,ap(ikp1+1),1)
                  call dcopy(km1,ap(ik+1),1,work,1)
                  ij = 0
                  do 200 j = 1, km1
                     jk = ik + j
                     ap(jk) = ddot(j,ap(ij+1),1,work,1)
                     call daxpy(j-1,work(j),ap(ij+1),1,ap(ik+1),1)
                     ij = ij + j
  200             continue
                  ap(kk) = ap(kk) + ddot(km1,work,1,ap(ik+1),1)
  210          continue
               kstep = 2
  220       continue
! 
! swap
! 
            ks = iabs(kpvt(k))
            if (ks .eq. k) go to 250
               iks = (ks*(ks - 1))/2
               call dswap(ks,ap(iks+1),1,ap(ik+1),1)
               ksj = ik + ks
               do 230 jb = ks, k
                  j = k + ks - jb
                  jk = ik + j
                  temp = ap(jk)
                  ap(jk) = ap(ksj)
                  ap(ksj) = temp
                  ksj = ksj - (j - 1)
  230          continue
               if (kstep .eq. 1) go to 240
                  kskp1 = ikp1 + ks
                  temp = ap(kskp1)
                  ap(kskp1) = ap(kkp1)
                  ap(kkp1) = temp
  240          continue
  250       continue
            ik = ik + k
            if (kstep .eq. 2) ik = ik + k + 1
            k = k + kstep
         go to 150
  260    continue
  270 continue
      return
      end
      subroutine dspfa(ap,n,kpvt,info)
      integer n,kpvt(1),info
      double precision ap(1)
! 
! dspfa factors a double precision symmetric matrix stored in
! packed form by elimination with symmetric pivoting.
! 
! to solve  a*x = b , follow dspfa by dspsl.
! to compute  inverse(a)*c , follow dspfa by dspsl.
! to compute  determinant(a) , follow dspfa by dspdi.
! to compute  inertia(a) , follow dspfa by dspdi.
! to compute  inverse(a) , follow dspfa by dspdi.
! 
! on entry
! 
! ap      double precision (n*(n+1)/2)
! the packed form of a symmetric matrix  a .  the
! columns of the upper triangle are stored sequentially
! in a one-dimensional array of length  n*(n+1)/2 .
! see comments below for details.
! 
! n       integer
! the order of the matrix  a .
! 
! output
! 
! ap      a block diagonal matrix and the multipliers which
! were used to obtain it stored in packed form.
! the factorization can be written  a = u*d*trans(u)
! where  u  is a product of permutation and unit
! upper triangular matrices , trans(u) is the
! transpose of  u , and  d  is block diagonal
! with 1 by 1 and 2 by 2 blocks.
! 
! kpvt    integer(n)
! an integer vector of pivot indices.
! 
! info    integer
! = 0  normal value.
! = k  if the k-th pivot block is singular. this is
! not an error condition for this subroutine,
! but it does indicate that dspsl or dspdi may
! divide by zero if called.
! 
! packed storage
! 
! the following program segment will pack the upper
! triangle of a symmetric matrix.
! 
! k = 0
! do 20 j = 1, n
! do 10 i = 1, j
! k = k + 1
! ap(k)  = a(i,j)
! 10    continue
! 20 continue
! 
! linpack. this version dated 08/14/78 .
! james bunch, univ. calif. san diego, argonne nat. lab.
! 
! subroutines and functions
! 
! blas daxpy,dswap,idamax
! fortran dabs,dmax1,dsqrt
! 
! internal variables
! 
      double precision ak,akm1,bk,bkm1,denom,mulk,mulkm1,t
      double precision absakk,alpha,colmax,rowmax
      integer idamax,ij,ijj,ik,ikm1,im,imax,imaxp1,imim,imj,imk
      integer j,jj,jk,jkm1,jmax,jmim,k,kk,km1,km1k,km1km1,km2,kstep
      logical swap
! 
! 
! initialize
! 
! alpha is used in choosing pivot block size.
      alpha = (1.0d0 + dsqrt(17.0d0))/8.0d0
! 
      info = 0
! 
! main loop on k, which goes from n to 1.
! 
      k = n
      ik = (n*(n - 1))/2
   10 continue
! 
! leave the loop if k=0 or k=1.
! 
! ...exit
         if (k .eq. 0) go to 200
         if (k .gt. 1) go to 20
            kpvt(1) = 1
            if (ap(1) .eq. 0.0d0) info = 1
! ......exit
            go to 200
   20    continue
! 
! this section of code determines the kind of
! elimination to be performed.  when it is completed,
! kstep will be set to the size of the pivot block, and
! swap will be set to .true. if an interchange is
! required.
! 
         km1 = k - 1
         kk = ik + k
         absakk = dabs(ap(kk))
! 
! determine the largest off-diagonal element in
! column k.
! 
         imax = idamax(k-1,ap(ik+1),1)
         imk = ik + imax
         colmax = dabs(ap(imk))
         if (absakk .lt. alpha*colmax) go to 30
            kstep = 1
            swap = .false.
         go to 90
   30    continue
! 
! determine the largest off-diagonal element in
! row imax.
! 
            rowmax = 0.0d0
            imaxp1 = imax + 1
            im = imax*(imax - 1)/2
            imj = im + 2*imax
            do 40 j = imaxp1, k
               rowmax = dmax1(rowmax,dabs(ap(imj)))
               imj = imj + j
   40       continue
            if (imax .eq. 1) go to 50
               jmax = idamax(imax-1,ap(im+1),1)
               jmim = jmax + im
               rowmax = dmax1(rowmax,dabs(ap(jmim)))
   50       continue
            imim = imax + im
            if (dabs(ap(imim)) .lt. alpha*rowmax) go to 60
               kstep = 1
               swap = .true.
            go to 80
   60       continue
            if (absakk .lt. alpha*colmax*(colmax/rowmax)) go to 70
               kstep = 1
               swap = .false.
            go to 80
   70       continue
               kstep = 2
               swap = imax .ne. km1
   80       continue
   90    continue
         if (dmax1(absakk,colmax) .ne. 0.0d0) go to 100
! 
! column k is zero.  set info and iterate the loop.
! 
            kpvt(k) = k
            info = k
         go to 190
  100    continue
         if (kstep .eq. 2) go to 140
! 
! 1 x 1 pivot block.
! 
            if (.not.swap) go to 120
! 
! perform an interchange.
! 
               call dswap(imax,ap(im+1),1,ap(ik+1),1)
               imj = ik + imax
               do 110 jj = imax, k
                  j = k + imax - jj
                  jk = ik + j
                  t = ap(jk)
                  ap(jk) = ap(imj)
                  ap(imj) = t
                  imj = imj - (j - 1)
  110          continue
  120       continue
! 
! perform the elimination.
! 
            ij = ik - (k - 1)
            do 130 jj = 1, km1
               j = k - jj
               jk = ik + j
               mulk = -ap(jk)/ap(kk)
               t = mulk
               call daxpy(j,t,ap(ik+1),1,ap(ij+1),1)
               ijj = ij + j
               ap(jk) = mulk
               ij = ij - (j - 1)
  130       continue
! 
! set the pivot array.
! 
            kpvt(k) = k
            if (swap) kpvt(k) = imax
         go to 190
  140    continue
! 
! 2 x 2 pivot block.
! 
            km1k = ik + k - 1
            ikm1 = ik - (k - 1)
            if (.not.swap) go to 160
! 
! perform an interchange.
! 
               call dswap(imax,ap(im+1),1,ap(ikm1+1),1)
               imj = ikm1 + imax
               do 150 jj = imax, km1
                  j = km1 + imax - jj
                  jkm1 = ikm1 + j
                  t = ap(jkm1)
                  ap(jkm1) = ap(imj)
                  ap(imj) = t
                  imj = imj - (j - 1)
  150          continue
               t = ap(km1k)
               ap(km1k) = ap(imk)
               ap(imk) = t
  160       continue
! 
! perform the elimination.
! 
            km2 = k - 2
            if (km2 .eq. 0) go to 180
               ak = ap(kk)/ap(km1k)
               km1km1 = ikm1 + k - 1
               akm1 = ap(km1km1)/ap(km1k)
               denom = 1.0d0 - ak*akm1
               ij = ik - (k - 1) - (k - 2)
               do 170 jj = 1, km2
                  j = km1 - jj
                  jk = ik + j
                  bk = ap(jk)/ap(km1k)
                  jkm1 = ikm1 + j
                  bkm1 = ap(jkm1)/ap(km1k)
                  mulk = (akm1*bk - bkm1)/denom
                  mulkm1 = (ak*bkm1 - bk)/denom
                  t = mulk
                  call daxpy(j,t,ap(ik+1),1,ap(ij+1),1)
                  t = mulkm1
                  call daxpy(j,t,ap(ikm1+1),1,ap(ij+1),1)
                  ap(jk) = mulk
                  ap(jkm1) = mulkm1
                  ijj = ij + j
                  ij = ij - (j - 1)
  170          continue
  180       continue
! 
! set the pivot array.
! 
            kpvt(k) = 1 - k
            if (swap) kpvt(k) = -imax
            kpvt(k-1) = kpvt(k)
  190    continue
         ik = ik - (k - 1)
         if (kstep .eq. 2) ik = ik - (k - 2)
         k = k - kstep
      go to 10
  200 continue
      return
      end
      double precision function pythag(a,b)
      double precision a,b
! 
! finds dsqrt(a**2+b**2) without overflow or destructive underflow
! 
      double precision p,r,s,t,u
      p = dmax1(dabs(a),dabs(b))
      if (p .eq. 0.0d0) go to 20
      r = (dmin1(dabs(a),dabs(b))/p)**2
   10 continue
         t = 4.0d0 + r
         if (t .eq. 4.0d0) go to 20
         s = r/t
         u = 1.0d0 + 2.0d0*s
         p = u*p
         r = (s/u)**2 * r
      go to 10
   20 pythag = p
      return
      end
      subroutine rs(nm,n,a,w,matz,z,fv1,fv2,ierr)
! 
      integer n,nm,ierr,matz
      double precision a(nm,n),w(n),z(nm,n),fv1(n),fv2(n)
! 
! this subroutine calls the recommended sequence of
! subroutines from the eigensystem subroutine package (eispack)
! to find the eigenvalues and eigenvectors (if desired)
! of a real symmetric matrix.
! 
! on input
! 
! nm  must be set to the row dimension of the two-dimensional
! array parameters as declared in the calling program
! dimension statement.
! 
! n  is the order of the matrix  a.
! 
! a  contains the real symmetric matrix.
! 
! matz  is an integer variable set equal to zero if
! only eigenvalues are desired.  otherwise it is set to
! any non-zero integer for both eigenvalues and eigenvectors.
! 
! on output
! 
! w  contains the eigenvalues in ascending order.
! 
! z  contains the eigenvectors if matz is not zero.
! 
! ierr  is an integer output variable set equal to an error
! completion code described in the documentation for tqlrat
! and tql2.  the normal completion code is zero.
! 
! fv1  and  fv2  are temporary storage arrays.
! 
! questions and comments should be directed to burton s. garbow,
! mathematics and computer science div, argonne national laboratory
! 
! this version dated august 1983.
! 
! ------------------------------------------------------------------
! 
      if (n .le. nm) go to 10
      ierr = 10 * n
      go to 50
! 
   10 if (matz .ne. 0) go to 20
! .......... find eigenvalues only ..........
      call  tred1(nm,n,a,w,fv1,fv2)
! tqlrat encounters catastrophic underflow on the Vax
! call  tqlrat(n,w,fv2,ierr)
      call  tql1(n,w,fv1,ierr)
      go to 50
! .......... find both eigenvalues and eigenvectors ..........
   20 call  tred2(nm,n,a,w,fv1,z)
      call  tql2(nm,n,w,fv1,z,ierr)
   50 return
      end
      subroutine tql1(n,d,e,ierr)
! 
      integer i,j,l,m,n,ii,l1,l2,mml,ierr
      double precision d(n),e(n)
      double precision c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2,pythag
! 
! this subroutine is a translation of the algol procedure tql1,
! num. math. 11, 293-306(1968) by bowdler, martin, reinsch, and
! wilkinson.
! handbook for auto. comp., vol.ii-linear algebra, 227-240(1971).
! 
! this subroutine finds the eigenvalues of a symmetric
! tridiagonal matrix by the ql method.
! 
! on input
! 
! n is the order of the matrix.
! 
! d contains the diagonal elements of the input matrix.
! 
! e contains the subdiagonal elements of the input matrix
! in its last n-1 positions.  e(1) is arbitrary.
! 
! on output
! 
! d contains the eigenvalues in ascending order.  if an
! error exit is made, the eigenvalues are correct and
! ordered for indices 1,2,...ierr-1, but may not be
! the smallest eigenvalues.
! 
! e has been destroyed.
! 
! ierr is set to
! zero       for normal return,
! j          if the j-th eigenvalue has not been
! determined after 30 iterations.
! 
! calls pythag for  dsqrt(a*a + b*b) .
! 
! questions and comments should be directed to burton s. garbow,
! mathematics and computer science div, argonne national laboratory
! 
! this version dated august 1983.
! 
! ------------------------------------------------------------------
! 
      ierr = 0
      if (n .eq. 1) go to 1001
! 
      do 100 i = 2, n
  100 e(i-1) = e(i)
! 
      f = 0.0d0
      tst1 = 0.0d0
      e(n) = 0.0d0
! 
      do 290 l = 1, n
         j = 0
         h = dabs(d(l)) + dabs(e(l))
         if (tst1 .lt. h) tst1 = h
! .......... look for small sub-diagonal element ..........
         do 110 m = l, n
            tst2 = tst1 + dabs(e(m))
            if (tst2 .eq. tst1) go to 120
! .......... e(n) is always zero, so there is no exit
! through the bottom of the loop ..........
  110    continue
! 
  120    if (m .eq. l) go to 210
  130    if (j .eq. 30) go to 1000
         j = j + 1
! .......... form shift ..........
         l1 = l + 1
         l2 = l1 + 1
         g = d(l)
         p = (d(l1) - g) / (2.0d0 * e(l))
         r = pythag(p,1.0d0)
         d(l) = e(l) / (p + dsign(r,p))
         d(l1) = e(l) * (p + dsign(r,p))
         dl1 = d(l1)
         h = g - d(l)
         if (l2 .gt. n) go to 145
! 
         do 140 i = l2, n
  140    d(i) = d(i) - h
! 
  145    f = f + h
! .......... ql transformation ..........
         p = d(m)
         c = 1.0d0
         c2 = c
         el1 = e(l1)
         s = 0.0d0
         mml = m - l
! .......... for i=m-1 step -1 until l do -- ..........
         do 200 ii = 1, mml
            c3 = c2
            c2 = c
            s2 = s
            i = m - ii
            g = c * e(i)
            h = c * p
            r = pythag(p,e(i))
            e(i+1) = s * r
            s = e(i) / r
            c = p / r
            p = c * d(i) - s * g
            d(i+1) = h + s * (c * g + s * d(i))
  200    continue
! 
         p = -s * s2 * c3 * el1 * e(l) / dl1
         e(l) = s * p
         d(l) = c * p
         tst2 = tst1 + dabs(e(l))
         if (tst2 .gt. tst1) go to 130
  210    p = d(l) + f
! .......... order eigenvalues ..........
         if (l .eq. 1) go to 250
! .......... for i=l step -1 until 2 do -- ..........
         do 230 ii = 2, l
            i = l + 2 - ii
            if (p .ge. d(i-1)) go to 270
            d(i) = d(i-1)
  230    continue
! 
  250    i = 1
  270    d(i) = p
  290 continue
! 
      go to 1001
! .......... set error -- no convergence to an
! eigenvalue after 30 iterations ..........
 1000 ierr = l
 1001 return
      end
      subroutine tql2(nm,n,d,e,z,ierr)
! 
      integer i,j,k,l,m,n,ii,l1,l2,nm,mml,ierr
      double precision d(n),e(n),z(nm,n)
      double precision c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2,pythag
! 
! this subroutine is a translation of the algol procedure tql2,
! num. math. 11, 293-306(1968) by bowdler, martin, reinsch, and
! wilkinson.
! handbook for auto. comp., vol.ii-linear algebra, 227-240(1971).
! 
! this subroutine finds the eigenvalues and eigenvectors
! of a symmetric tridiagonal matrix by the ql method.
! the eigenvectors of a full symmetric matrix can also
! be found if  tred2  has been used to reduce this
! full matrix to tridiagonal form.
! 
! on input
! 
! nm must be set to the row dimension of two-dimensional
! array parameters as declared in the calling program
! dimension statement.
! 
! n is the order of the matrix.
! 
! d contains the diagonal elements of the input matrix.
! 
! e contains the subdiagonal elements of the input matrix
! in its last n-1 positions.  e(1) is arbitrary.
! 
! z contains the transformation matrix produced in the
! reduction by  tred2, if performed.  if the eigenvectors
! of the tridiagonal matrix are desired, z must contain
! the identity matrix.
! 
! on output
! 
! d contains the eigenvalues in ascending order.  if an
! error exit is made, the eigenvalues are correct but
! unordered for indices 1,2,...,ierr-1.
! 
! e has been destroyed.
! 
! z contains orthonormal eigenvectors of the symmetric
! tridiagonal (or full) matrix.  if an error exit is made,
! z contains the eigenvectors associated with the stored
! eigenvalues.
! 
! ierr is set to
! zero       for normal return,
! j          if the j-th eigenvalue has not been
! determined after 30 iterations.
! 
! calls pythag for  dsqrt(a*a + b*b) .
! 
! questions and comments should be directed to burton s. garbow,
! mathematics and computer science div, argonne national laboratory
! 
! this version dated august 1983.
! 
! ------------------------------------------------------------------
! 
      ierr = 0
      if (n .eq. 1) go to 1001
! 
      do 100 i = 2, n
  100 e(i-1) = e(i)
! 
      f = 0.0d0
      tst1 = 0.0d0
      e(n) = 0.0d0
! 
      do 240 l = 1, n
         j = 0
         h = dabs(d(l)) + dabs(e(l))
         if (tst1 .lt. h) tst1 = h
! .......... look for small sub-diagonal element ..........
         do 110 m = l, n
            tst2 = tst1 + dabs(e(m))
            if (tst2 .eq. tst1) go to 120
! .......... e(n) is always zero, so there is no exit
! through the bottom of the loop ..........
  110    continue
! 
  120    if (m .eq. l) go to 220
  130    if (j .eq. 30) go to 1000
         j = j + 1
! .......... form shift ..........
         l1 = l + 1
         l2 = l1 + 1
         g = d(l)
         p = (d(l1) - g) / (2.0d0 * e(l))
         r = pythag(p,1.0d0)
         d(l) = e(l) / (p + dsign(r,p))
         d(l1) = e(l) * (p + dsign(r,p))
         dl1 = d(l1)
         h = g - d(l)
         if (l2 .gt. n) go to 145
! 
         do 140 i = l2, n
  140    d(i) = d(i) - h
! 
  145    f = f + h
! .......... ql transformation ..........
         p = d(m)
         c = 1.0d0
         c2 = c
         el1 = e(l1)
         s = 0.0d0
         mml = m - l
! .......... for i=m-1 step -1 until l do -- ..........
         do 200 ii = 1, mml
            c3 = c2
            c2 = c
            s2 = s
            i = m - ii
            g = c * e(i)
            h = c * p
            r = pythag(p,e(i))
            e(i+1) = s * r
            s = e(i) / r
            c = p / r
            p = c * d(i) - s * g
            d(i+1) = h + s * (c * g + s * d(i))
! .......... form vector ..........
            do 180 k = 1, n
               h = z(k,i+1)
               z(k,i+1) = s * z(k,i) + c * h
               z(k,i) = c * z(k,i) - s * h
  180       continue
! 
  200    continue
! 
         p = -s * s2 * c3 * el1 * e(l) / dl1
         e(l) = s * p
         d(l) = c * p
         tst2 = tst1 + dabs(e(l))
         if (tst2 .gt. tst1) go to 130
  220    d(l) = d(l) + f
  240 continue
! .......... order eigenvalues and eigenvectors ..........
      do 300 ii = 2, n
         i = ii - 1
         k = i
         p = d(i)
! 
         do 260 j = ii, n
            if (d(j) .ge. p) go to 260
            k = j
            p = d(j)
  260    continue
! 
         if (k .eq. i) go to 300
         d(k) = d(i)
         d(i) = p
! 
         do 280 j = 1, n
            p = z(j,i)
            z(j,i) = z(j,k)
            z(j,k) = p
  280    continue
! 
  300 continue
! 
      go to 1001
! .......... set error -- no convergence to an
! eigenvalue after 30 iterations ..........
 1000 ierr = l
 1001 return
      end
      subroutine tred1(nm,n,a,d,e,e2)
! 
      integer i,j,k,l,n,ii,nm,jp1
      double precision a(nm,n),d(n),e(n),e2(n)
      double precision f,g,h,scale
! 
! this subroutine is a translation of the algol procedure tred1,
! num. math. 11, 181-195(1968) by martin, reinsch, and wilkinson.
! handbook for auto. comp., vol.ii-linear algebra, 212-226(1971).
! 
! this subroutine reduces a real symmetric matrix
! to a symmetric tridiagonal matrix using
! orthogonal similarity transformations.
! 
! on input
! 
! nm must be set to the row dimension of two-dimensional
! array parameters as declared in the calling program
! dimension statement.
! 
! n is the order of the matrix.
! 
! a contains the real symmetric input matrix.  only the
! lower triangle of the matrix need be supplied.
! 
! on output
! 
! a contains information about the orthogonal trans-
! formations used in the reduction in its strict lower
! triangle.  the full upper triangle of a is unaltered.
! 
! d contains the diagonal elements of the tridiagonal matrix.
! 
! e contains the subdiagonal elements of the tridiagonal
! matrix in its last n-1 positions.  e(1) is set to zero.
! 
! e2 contains the squares of the corresponding elements of e.
! e2 may coincide with e if the squares are not needed.
! 
! questions and comments should be directed to burton s. garbow,
! mathematics and computer science div, argonne national laboratory
! 
! this version dated august 1983.
! 
! ------------------------------------------------------------------
! 
      do 100 i = 1, n
         d(i) = a(n,i)
         a(n,i) = a(i,i)
  100 continue
! .......... for i=n step -1 until 1 do -- ..........
      do 300 ii = 1, n
         i = n + 1 - ii
         l = i - 1
         h = 0.0d0
         scale = 0.0d0
         if (l .lt. 1) go to 130
! .......... scale row (algol tol then not needed) ..........
         do 120 k = 1, l
  120    scale = scale + dabs(d(k))
! 
         if (scale .ne. 0.0d0) go to 140
! 
         do 125 j = 1, l
            d(j) = a(l,j)
            a(l,j) = a(i,j)
            a(i,j) = 0.0d0
  125    continue
! 
  130    e(i) = 0.0d0
         e2(i) = 0.0d0
         go to 300
! 
  140    do 150 k = 1, l
            d(k) = d(k) / scale
            h = h + d(k) * d(k)
  150    continue
! 
         e2(i) = scale * scale * h
         f = d(l)
         g = -dsign(dsqrt(h),f)
         e(i) = scale * g
         h = h - f * g
         d(l) = f - g
         if (l .eq. 1) go to 285
! .......... form a*u ..........
         do 170 j = 1, l
  170    e(j) = 0.0d0
! 
         do 240 j = 1, l
            f = d(j)
            g = e(j) + a(j,j) * f
            jp1 = j + 1
            if (l .lt. jp1) go to 220
! 
            do 200 k = jp1, l
               g = g + a(k,j) * d(k)
               e(k) = e(k) + a(k,j) * f
  200       continue
! 
  220       e(j) = g
  240    continue
! .......... form p ..........
         f = 0.0d0
! 
         do 245 j = 1, l
            e(j) = e(j) / h
            f = f + e(j) * d(j)
  245    continue
! 
         h = f / (h + h)
! .......... form q ..........
         do 250 j = 1, l
  250    e(j) = e(j) - h * d(j)
! .......... form reduced a ..........
         do 280 j = 1, l
            f = d(j)
            g = e(j)
! 
            do 260 k = j, l
  260       a(k,j) = a(k,j) - f * e(k) - g * d(k)
! 
  280    continue
! 
  285    do 290 j = 1, l
            f = d(j)
            d(j) = a(l,j)
            a(l,j) = a(i,j)
            a(i,j) = f * scale
  290    continue
! 
  300 continue
! 
      return
      end
      subroutine tred2(nm,n,a,d,e,z)
! 
      integer i,j,k,l,n,ii,nm,jp1
      double precision a(nm,n),d(n),e(n),z(nm,n)
      double precision f,g,h,hh,scale
! 
! this subroutine is a translation of the algol procedure tred2,
! num. math. 11, 181-195(1968) by martin, reinsch, and wilkinson.
! handbook for auto. comp., vol.ii-linear algebra, 212-226(1971).
! 
! this subroutine reduces a real symmetric matrix to a
! symmetric tridiagonal matrix using and accumulating
! orthogonal similarity transformations.
! 
! on input
! 
! nm must be set to the row dimension of two-dimensional
! array parameters as declared in the calling program
! dimension statement.
! 
! n is the order of the matrix.
! 
! a contains the real symmetric input matrix.  only the
! lower triangle of the matrix need be supplied.
! 
! on output
! 
! d contains the diagonal elements of the tridiagonal matrix.
! 
! e contains the subdiagonal elements of the tridiagonal
! matrix in its last n-1 positions.  e(1) is set to zero.
! 
! z contains the orthogonal transformation matrix
! produced in the reduction.
! 
! a and z may coincide.  if distinct, a is unaltered.
! 
! questions and comments should be directed to burton s. garbow,
! mathematics and computer science div, argonne national laboratory
! 
! this version dated august 1983.
! 
! ------------------------------------------------------------------
! 
      do 100 i = 1, n
! 
         do 80 j = i, n
   80    z(j,i) = a(j,i)
! 
         d(i) = a(n,i)
  100 continue
! 
      if (n .eq. 1) go to 510
! .......... for i=n step -1 until 2 do -- ..........
      do 300 ii = 2, n
         i = n + 2 - ii
         l = i - 1
         h = 0.0d0
         scale = 0.0d0
         if (l .lt. 2) go to 130
! .......... scale row (algol tol then not needed) ..........
         do 120 k = 1, l
  120    scale = scale + dabs(d(k))
! 
         if (scale .ne. 0.0d0) go to 140
  130    e(i) = d(l)
! 
         do 135 j = 1, l
            d(j) = z(l,j)
            z(i,j) = 0.0d0
            z(j,i) = 0.0d0
  135    continue
! 
         go to 290
! 
  140    do 150 k = 1, l
            d(k) = d(k) / scale
            h = h + d(k) * d(k)
  150    continue
! 
         f = d(l)
         g = -dsign(dsqrt(h),f)
         e(i) = scale * g
         h = h - f * g
         d(l) = f - g
! .......... form a*u ..........
         do 170 j = 1, l
  170    e(j) = 0.0d0
! 
         do 240 j = 1, l
            f = d(j)
            z(j,i) = f
            g = e(j) + z(j,j) * f
            jp1 = j + 1
            if (l .lt. jp1) go to 220
! 
            do 200 k = jp1, l
               g = g + z(k,j) * d(k)
               e(k) = e(k) + z(k,j) * f
  200       continue
! 
  220       e(j) = g
  240    continue
! .......... form p ..........
         f = 0.0d0
! 
         do 245 j = 1, l
            e(j) = e(j) / h
            f = f + e(j) * d(j)
  245    continue
! 
         hh = f / (h + h)
! .......... form q ..........
         do 250 j = 1, l
  250    e(j) = e(j) - hh * d(j)
! .......... form reduced a ..........
         do 280 j = 1, l
            f = d(j)
            g = e(j)
! 
            do 260 k = j, l
  260       z(k,j) = z(k,j) - f * e(k) - g * d(k)
! 
            d(j) = z(l,j)
            z(i,j) = 0.0d0
  280    continue
! 
  290    d(i) = h
  300 continue
! .......... accumulation of transformation matrices ..........
      do 500 i = 2, n
         l = i - 1
         z(n,l) = z(l,l)
         z(l,l) = 1.0d0
         h = d(i)
         if (h .eq. 0.0d0) go to 380
! 
         do 330 k = 1, l
  330    d(k) = z(k,i) / h
! 
         do 360 j = 1, l
            g = 0.0d0
! 
            do 340 k = 1, l
  340       g = g + z(k,i) * z(k,j)
! 
            do 360 k = 1, l
               z(k,j) = z(k,j) - g * d(k)
  360    continue
! 
  380    do 400 k = 1, l
  400    z(k,i) = 0.0d0
! 
  500 continue
! 
  510 do 520 i = 1, n
         d(i) = z(n,i)
         z(n,i) = 0.0d0
  520 continue
! 
      z(n,n) = 1.0d0
      e(1) = 0.0d0
      return
      end
      subroutine daxpy(n,da,dx,incx,dy,incy)
! 
! constant times a vector plus a vector.
! uses unrolled loops for increments equal to one.
! jack dongarra, linpack, 3/11/78.
! modified 12/3/93, array(1) declarations changed to array(*)
! 
      double precision dx(*),dy(*),da
      integer i,incx,incy,ix,iy,m,mp1,n
! 
      if(n.le.0)return
      if (da .eq. 0.0d0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
! 
! code for unequal increments or equal increments
! not equal to 1
! 
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dy(iy) + da*dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
! 
! code for both increments equal to 1
! 
! 
! clean-up loop
! 
   20 m = mod(n,4)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dy(i) + da*dx(i)
   30 continue
      if( n .lt. 4 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,4
        dy(i) = dy(i) + da*dx(i)
        dy(i + 1) = dy(i + 1) + da*dx(i + 1)
        dy(i + 2) = dy(i + 2) + da*dx(i + 2)
        dy(i + 3) = dy(i + 3) + da*dx(i + 3)
   50 continue
      return
      end
      subroutine  dcopy(n,dx,incx,dy,incy)
! 
! copies a vector, x, to a vector, y.
! uses unrolled loops for increments equal to one.
! jack dongarra, linpack, 3/11/78.
! modified 12/3/93, array(1) declarations changed to array(*)
! 
      double precision dx(*),dy(*)
      integer i,incx,incy,ix,iy,m,mp1,n
! 
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
! 
! code for unequal increments or equal increments
! not equal to 1
! 
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
! 
! code for both increments equal to 1
! 
! 
! clean-up loop
! 
   20 m = mod(n,7)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dx(i)
   30 continue
      if( n .lt. 7 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,7
        dy(i) = dx(i)
        dy(i + 1) = dx(i + 1)
        dy(i + 2) = dx(i + 2)
        dy(i + 3) = dx(i + 3)
        dy(i + 4) = dx(i + 4)
        dy(i + 5) = dx(i + 5)
        dy(i + 6) = dx(i + 6)
   50 continue
      return
      end
      double precision function ddot(n,dx,incx,dy,incy)
! 
! forms the dot product of two vectors.
! uses unrolled loops for increments equal to one.
! jack dongarra, linpack, 3/11/78.
! modified 12/3/93, array(1) declarations changed to array(*)
! 
      double precision dx(*),dy(*),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
! 
      ddot = 0.0d0
      dtemp = 0.0d0
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
! 
! code for unequal increments or equal increments
! not equal to 1
! 
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dtemp + dx(ix)*dy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      ddot = dtemp
      return
! 
! code for both increments equal to 1
! 
! 
! clean-up loop
! 
   20 m = mod(n,5)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dtemp + dx(i)*dy(i)
   30 continue
      if( n .lt. 5 ) go to 60
   40 mp1 = m + 1
      do 50 i = mp1,n,5
        dtemp = dtemp + dx(i)*dy(i) + dx(i + 1)*dy(i + 1) +
     *   dx(i + 2)*dy(i + 2) + dx(i + 3)*dy(i + 3) + dx(i + 4)*dy(i + 4)
   50 continue
   60 ddot = dtemp
      return
      end
      subroutine  dswap (n,dx,incx,dy,incy)
! 
! interchanges two vectors.
! uses unrolled loops for increments equal one.
! jack dongarra, linpack, 3/11/78.
! modified 12/3/93, array(1) declarations changed to array(*)
! 
      double precision dx(*),dy(*),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
! 
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
! 
! code for unequal increments or equal increments not equal
! to 1
! 
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dx(ix)
        dx(ix) = dy(iy)
        dy(iy) = dtemp
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
! 
! code for both increments equal to 1
! 
! 
! clean-up loop
! 
   20 m = mod(n,3)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dx(i)
        dx(i) = dy(i)
        dy(i) = dtemp
   30 continue
      if( n .lt. 3 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,3
        dtemp = dx(i)
        dx(i) = dy(i)
        dy(i) = dtemp
        dtemp = dx(i + 1)
        dx(i + 1) = dy(i + 1)
        dy(i + 1) = dtemp
        dtemp = dx(i + 2)
        dx(i + 2) = dy(i + 2)
        dy(i + 2) = dtemp
   50 continue
      return
      end
      integer function idamax(n,dx,incx)
! 
! finds the index of element having max. absolute value.
! jack dongarra, linpack, 3/11/78.
! modified 3/93 to return if incx .le. 0.
! modified 12/3/93, array(1) declarations changed to array(*)
! 
      double precision dx(*),dmax
      integer i,incx,ix,n
! 
      idamax = 0
      if( n.lt.1 .or. incx.le.0 ) return
      idamax = 1
      if(n.eq.1)return
      if(incx.eq.1)go to 20
! 
! code for increment not equal to 1
! 
      ix = 1
      dmax = dabs(dx(1))
      ix = ix + incx
      do 10 i = 2,n
         if(dabs(dx(ix)).le.dmax) go to 5
         idamax = i
         dmax = dabs(dx(ix))
    5    ix = ix + incx
   10 continue
      return
! 
! code for increment equal to 1
! 
   20 dmax = dabs(dx(1))
      do 30 i = 2,n
         if(dabs(dx(i)).le.dmax) go to 30
         idamax = i
         dmax = dabs(dx(i))
   30 continue
      return
      end
      SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
! .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
! ..
! .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
! ..
! 
! Purpose
! =======
! 
! DGEMM  performs one of the matrix-matrix operations
! 
! C := alpha*op( A )*op( B ) + beta*C,
! 
! where  op( X ) is one of
! 
! op( X ) = X   or   op( X ) = X',
! 
! alpha and beta are scalars, and A, B and C are matrices, with op( A )
! an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
! 
! Arguments
! ==========
! 
! TRANSA - CHARACTER*1.
! On entry, TRANSA specifies the form of op( A ) to be used in
! the matrix multiplication as follows:
! 
! TRANSA = 'N' or 'n',  op( A ) = A.
! 
! TRANSA = 'T' or 't',  op( A ) = A'.
! 
! TRANSA = 'C' or 'c',  op( A ) = A'.
! 
! Unchanged on exit.
! 
! TRANSB - CHARACTER*1.
! On entry, TRANSB specifies the form of op( B ) to be used in
! the matrix multiplication as follows:
! 
! TRANSB = 'N' or 'n',  op( B ) = B.
! 
! TRANSB = 'T' or 't',  op( B ) = B'.
! 
! TRANSB = 'C' or 'c',  op( B ) = B'.
! 
! Unchanged on exit.
! 
! M      - INTEGER.
! On entry,  M  specifies  the number  of rows  of the  matrix
! op( A )  and of the  matrix  C.  M  must  be at least  zero.
! Unchanged on exit.
! 
! N      - INTEGER.
! On entry,  N  specifies the number  of columns of the matrix
! op( B ) and the number of columns of the matrix C. N must be
! at least zero.
! Unchanged on exit.
! 
! K      - INTEGER.
! On entry,  K  specifies  the number of columns of the matrix
! op( A ) and the number of rows of the matrix op( B ). K must
! be at least  zero.
! Unchanged on exit.
! 
! ALPHA  - DOUBLE PRECISION.
! On entry, ALPHA specifies the scalar alpha.
! Unchanged on exit.
! 
! A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
! k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
! Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
! part of the array  A  must contain the matrix  A,  otherwise
! the leading  k by m  part of the array  A  must contain  the
! matrix A.
! Unchanged on exit.
! 
! LDA    - INTEGER.
! On entry, LDA specifies the first dimension of A as declared
! in the calling (sub) program. When  TRANSA = 'N' or 'n' then
! LDA must be at least  max( 1, m ), otherwise  LDA must be at
! least  max( 1, k ).
! Unchanged on exit.
! 
! B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
! n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
! Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
! part of the array  B  must contain the matrix  B,  otherwise
! the leading  n by k  part of the array  B  must contain  the
! matrix B.
! Unchanged on exit.
! 
! LDB    - INTEGER.
! On entry, LDB specifies the first dimension of B as declared
! in the calling (sub) program. When  TRANSB = 'N' or 'n' then
! LDB must be at least  max( 1, k ), otherwise  LDB must be at
! least  max( 1, n ).
! Unchanged on exit.
! 
! BETA   - DOUBLE PRECISION.
! On entry,  BETA  specifies the scalar  beta.  When  BETA  is
! supplied as zero then C need not be set on input.
! Unchanged on exit.
! 
! C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
! Before entry, the leading  m by n  part of the array  C must
! contain the matrix  C,  except when  beta  is zero, in which
! case C need not be set on entry.
! On exit, the array  C  is overwritten by the  m by n  matrix
! ( alpha*op( A )*op( B ) + beta*C ).
! 
! LDC    - INTEGER.
! On entry, LDC specifies the first dimension of C as declared
! in  the  calling  (sub)  program.   LDC  must  be  at  least
! max( 1, m ).
! Unchanged on exit.
! 
! 
! Level 3 Blas routine.
! 
! -- Written on 8-February-1989.
! Jack Dongarra, Argonne National Laboratory.
! Iain Duff, AERE Harwell.
! Jeremy Du Croz, Numerical Algorithms Group Ltd.
! Sven Hammarling, Numerical Algorithms Group Ltd.
! 
! 
! .. External Functions ..
      LOGICAL LSAME
      EXTERNAL LSAME
! ..
! .. External Subroutines ..
      EXTERNAL XERBLA
! ..
! .. Intrinsic Functions ..
      INTRINSIC MAX
! ..
! .. Local Scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I,INFO,J,L,NCOLA,NROWA,NROWB
      LOGICAL NOTA,NOTB
! ..
! .. Parameters ..
      DOUBLE PRECISION ONE,ZERO
      PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
! ..
! 
! Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
! transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
! and  columns of  A  and the  number of  rows  of  B  respectively.
! 
      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      IF (NOTA) THEN
          NROWA = M
          NCOLA = K
      ELSE
          NROWA = K
          NCOLA = M
      END IF
      IF (NOTB) THEN
          NROWB = K
      ELSE
          NROWB = N
      END IF
! 
! Test the input parameters.
! 
      INFO = 0
      IF ((.NOT.NOTA) .AND. (.NOT.LSAME(TRANSA,'C')) .AND.
     +    (.NOT.LSAME(TRANSA,'T'))) THEN
          INFO = 1
      ELSE IF ((.NOT.NOTB) .AND. (.NOT.LSAME(TRANSB,'C')) .AND.
     +         (.NOT.LSAME(TRANSB,'T'))) THEN
          INFO = 2
      ELSE IF (M.LT.0) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (K.LT.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,NROWA)) THEN
          INFO = 8
      ELSE IF (LDB.LT.MAX(1,NROWB)) THEN
          INFO = 10
      ELSE IF (LDC.LT.MAX(1,M)) THEN
          INFO = 13
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMM ',INFO)
          RETURN
      END IF
! 
! Quick return if possible.
! 
      IF ((M.EQ.0) .OR. (N.EQ.0) .OR.
     +    (((ALPHA.EQ.ZERO).OR. (K.EQ.0)).AND. (BETA.EQ.ONE))) RETURN
! 
! And if  alpha.eq.zero.
! 
      IF (ALPHA.EQ.ZERO) THEN
          IF (BETA.EQ.ZERO) THEN
              DO 20 J = 1,N
                  DO 10 I = 1,M
                      C(I,J) = ZERO
   10             CONTINUE
   20         CONTINUE
          ELSE
              DO 40 J = 1,N
                  DO 30 I = 1,M
                      C(I,J) = BETA*C(I,J)
   30             CONTINUE
   40         CONTINUE
          END IF
          RETURN
      END IF
! 
! Start the operations.
! 
      IF (NOTB) THEN
          IF (NOTA) THEN
! 
! Form  C := alpha*A*B + beta*C.
! 
              DO 90 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 50 I = 1,M
                          C(I,J) = ZERO
   50                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 60 I = 1,M
                          C(I,J) = BETA*C(I,J)
   60                 CONTINUE
                  END IF
                  DO 80 L = 1,K
                      IF (B(L,J).NE.ZERO) THEN
                          TEMP = ALPHA*B(L,J)
                          DO 70 I = 1,M
                              C(I,J) = C(I,J) + TEMP*A(I,L)
   70                     CONTINUE
                      END IF
   80             CONTINUE
   90         CONTINUE
          ELSE
! 
! Form  C := alpha*A'*B + beta*C
! 
              DO 120 J = 1,N
                  DO 110 I = 1,M
                      TEMP = ZERO
                      DO 100 L = 1,K
                          TEMP = TEMP + A(L,I)*B(L,J)
  100                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  110             CONTINUE
  120         CONTINUE
          END IF
      ELSE
          IF (NOTA) THEN
! 
! Form  C := alpha*A*B' + beta*C
! 
              DO 170 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 130 I = 1,M
                          C(I,J) = ZERO
  130                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 140 I = 1,M
                          C(I,J) = BETA*C(I,J)
  140                 CONTINUE
                  END IF
                  DO 160 L = 1,K
                      IF (B(J,L).NE.ZERO) THEN
                          TEMP = ALPHA*B(J,L)
                          DO 150 I = 1,M
                              C(I,J) = C(I,J) + TEMP*A(I,L)
  150                     CONTINUE
                      END IF
  160             CONTINUE
  170         CONTINUE
          ELSE
! 
! Form  C := alpha*A'*B' + beta*C
! 
              DO 200 J = 1,N
                  DO 190 I = 1,M
                      TEMP = ZERO
                      DO 180 L = 1,K
                          TEMP = TEMP + A(L,I)*B(J,L)
  180                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  190             CONTINUE
  200         CONTINUE
          END IF
      END IF
! 
      RETURN
! 
! End of DGEMM .
! 
      END
      LOGICAL          FUNCTION LSAME( CA, CB )
! ***************************************************************************
! *
! DATA PARALLEL BLAS based on MPL                                        *
! *
! Version 1.0   1/9-92 ,                                                 *
! For MasPar MP-1 computers                                              *
! *
! para//ab, University of Bergen, NORWAY                                 *
! *
! These programs must be called using F90 style array syntax.            *
! Note that the F77 style calling sequence has been retained             *
! in this version for compatibility reasons, be aware that               *
! parameters related to the array dimensions and shape therefore may     *
! be redundant and without any influence.                                *
! The calling sequence may be changed in a future version.               *
! Please report any BUGs, ideas for improvement or other                 *
! comments to                                                            *
! adm@parallab.uib.no                                   *
! *
! Future versions may then reflect your suggestions.                     *
! The most current version of this software is available                 *
! from netlib@nac.no , send the message `send index from maspar'         *
! *
! REVISIONS:                                                             *
! *
! ***************************************************************************
! 
! -- LAPACK auxiliary routine (preliminary version) --
! Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
! Courant Institute, NAG Ltd., and Rice University
! March 26, 1990
! 
! .. Scalar Arguments ..
      CHARACTER          CA, CB
! ..
! 
! Purpose
! =======
! 
! LSAME returns .TRUE. if CA is the same letter as CB regardless of
! case.
! 
! This version of the routine is only correct for ASCII code.
! Installers must modify the routine for other character-codes.
! 
! For EBCDIC systems the constant IOFF must be changed to -64.
! For CDC systems using 6-12 bit representations, the system-
! specific code in comments must be activated.
! 
! Arguments
! =========
! 
! CA      (input) CHARACTER*1
! CB      (input) CHARACTER*1
! CA and CB specify the single characters to be compared.
! 
! 
! .. Parameters ..
      INTEGER            IOFF
      PARAMETER        ( IOFF = 32 )
! ..
! .. Intrinsic Functions ..
      INTRINSIC          ICHAR
! ..
! .. Executable Statements ..
! 
! Test if the characters are equal
! 
      LSAME = CA.EQ.CB
! 
! Now test for equivalence
! 
      IF( .NOT.LSAME ) THEN
         LSAME = ICHAR( CA ) - IOFF.EQ.ICHAR( CB )
      END IF
      IF( .NOT.LSAME ) THEN
         LSAME = ICHAR( CA ).EQ.ICHAR( CB ) - IOFF
      END IF
! 
      RETURN
! 
! The following comments contain code for CDC systems using 6-12 bit
! representations.
! 
! .. Parameters ..
! INTEGER            ICIRFX
! PARAMETER        ( ICIRFX=62 )
! .. Scalar arguments ..
! CHARACTER*1        CB
! .. Array arguments ..
! CHARACTER*1        CA(*)
! .. Local scalars ..
! INTEGER            IVAL
! .. Intrinsic functions ..
! INTRINSIC          ICHAR, CHAR
! .. Executable statements ..
! 
! See if the first character in string CA equals string CB.
! 
! LSAME = CA(1) .EQ. CB .AND. CA(1) .NE. CHAR(ICIRFX)
! 
! IF (LSAME) RETURN
! 
! The characters are not identical. Now check them for equivalence.
! Look for the 'escape' character, circumflex, followed by the
! letter.
! 
! IVAL = ICHAR(CA(2))
! IF (IVAL.GE.ICHAR('A') .AND. IVAL.LE.ICHAR('Z')) THEN
! LSAME = CA(1) .EQ. CHAR(ICIRFX) .AND. CA(2) .EQ. CB
! END IF
! 
! RETURN
! 
! End of LSAME
! 
      END
      SUBROUTINE XERBLA( SRNAME, INFO )
! ***************************************************************************
! *
! DATA PARALLEL BLAS based on MPL                                        *
! *
! Version 1.0   1/9-92 ,                                                 *
! For MasPar MP-1 computers                                              *
! *
! para//ab, University of Bergen, NORWAY                                 *
! *
! These programs must be called using F90 style array syntax.            *
! Note that the F77 style calling sequence has been retained             *
! in this version for compatibility reasons, be aware that               *
! parameters related to the array dimensions and shape therefore may     *
! be redundant and without any influence.                                *
! The calling sequence may be changed in a future version.               *
! Please report any BUGs, ideas for improvement or other                 *
! comments to                                                            *
! adm@parallab.uib.no                                   *
! *
! Future versions may then reflect your suggestions.                     *
! The most current version of this software is available                 *
! from netlib@nac.no , send the message `send index from maspar'         *
! *
! REVISIONS:                                                             *
! *
! ***************************************************************************
! 
! -- LAPACK auxiliary routine (preliminary version) --
! Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
! Courant Institute, NAG Ltd., and Rice University
! March 26, 1990
! 
! .. Scalar Arguments ..
      CHARACTER*6        SRNAME
      INTEGER            INFO
! ..
! 
! Purpose
! =======
! 
! XERBLA  is an error handler for the LAPACK routines.
! It is called by an LAPACK routine if an input parameter has an
! invalid value.  A message is printed and execution stops.
! 
! Installers may consider modifying the STOP statement in order to
! call system-specific exception-handling facilities.
! 
! Arguments
! =========
! 
! SRNAME  (input) CHARACTER*6
! The name of the routine which called XERBLA.
! 
! INFO    (input) INTEGER
! The position of the invalid parameter in the parameter list
! of the calling routine.
! 
! 
      WRITE( *, FMT = 9999 )SRNAME, INFO
! 
      STOP
! 
 9999 FORMAT( ' ** On entry to ', A6, ' parameter number ', I2, ' had ',
     $      'an illegal value' )
! 
! End of XERBLA
! 
      END


      SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
! .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER INCX,INCY,LDA,M,N
      CHARACTER TRANS
! ..
! .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
! ..
! 
! Purpose
! =======
! 
! DGEMV  performs one of the matrix-vector operations
! 
! y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
! 
! where alpha and beta are scalars, x and y are vectors and A is an
! m by n matrix.
! 
! Arguments
! ==========
! 
! TRANS  - CHARACTER*1.
! On entry, TRANS specifies the operation to be performed as
! follows:
! 
! TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
! 
! TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
! 
! TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.
! 
! Unchanged on exit.
! 
! M      - INTEGER.
! On entry, M specifies the number of rows of the matrix A.
! M must be at least zero.
! Unchanged on exit.
! 
! N      - INTEGER.
! On entry, N specifies the number of columns of the matrix A.
! N must be at least zero.
! Unchanged on exit.
! 
! ALPHA  - DOUBLE PRECISION.
! On entry, ALPHA specifies the scalar alpha.
! Unchanged on exit.
! 
! A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
! Before entry, the leading m by n part of the array A must
! contain the matrix of coefficients.
! Unchanged on exit.
! 
! LDA    - INTEGER.
! On entry, LDA specifies the first dimension of A as declared
! in the calling (sub) program. LDA must be at least
! max( 1, m ).
! Unchanged on exit.
! 
! X      - DOUBLE PRECISION array of DIMENSION at least
! ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
! and at least
! ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
! Before entry, the incremented array X must contain the
! vector x.
! Unchanged on exit.
! 
! INCX   - INTEGER.
! On entry, INCX specifies the increment for the elements of
! X. INCX must not be zero.
! Unchanged on exit.
! 
! BETA   - DOUBLE PRECISION.
! On entry, BETA specifies the scalar beta. When BETA is
! supplied as zero then Y need not be set on input.
! Unchanged on exit.
! 
! Y      - DOUBLE PRECISION array of DIMENSION at least
! ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
! and at least
! ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
! Before entry with BETA non-zero, the incremented array Y
! must contain the vector y. On exit, Y is overwritten by the
! updated vector y.
! 
! INCY   - INTEGER.
! On entry, INCY specifies the increment for the elements of
! Y. INCY must not be zero.
! Unchanged on exit.
! 
! 
! Level 2 Blas routine.
! 
! -- Written on 22-October-1986.
! Jack Dongarra, Argonne National Lab.
! Jeremy Du Croz, Nag Central Office.
! Sven Hammarling, Nag Central Office.
! Richard Hanson, Sandia National Labs.
! 
! 
! .. Parameters ..
      DOUBLE PRECISION ONE,ZERO
      PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
! ..
! .. Local Scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I,INFO,IX,IY,J,JX,JY,KX,KY,LENX,LENY
! ..
! .. External Functions ..
      LOGICAL LSAME
      EXTERNAL LSAME
! ..
! .. External Subroutines ..
      EXTERNAL XERBLA
! ..
! .. Intrinsic Functions ..
      INTRINSIC MAX
! ..
! 
! Test the input parameters.
! 
      INFO = 0
      IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND.
     +    .NOT.LSAME(TRANS,'C')) THEN
          INFO = 1
      ELSE IF (M.LT.0) THEN
          INFO = 2
      ELSE IF (N.LT.0) THEN
          INFO = 3
      ELSE IF (LDA.LT.MAX(1,M)) THEN
          INFO = 6
      ELSE IF (INCX.EQ.0) THEN
          INFO = 8
      ELSE IF (INCY.EQ.0) THEN
          INFO = 11
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMV ',INFO)
          RETURN
      END IF
! 
! Quick return if possible.
! 
      IF ((M.EQ.0) .OR. (N.EQ.0) .OR.
     +    ((ALPHA.EQ.ZERO).AND. (BETA.EQ.ONE))) RETURN
! 
! Set  LENX  and  LENY, the lengths of the vectors x and y, and set
! up the start points in  X  and  Y.
! 
      IF (LSAME(TRANS,'N')) THEN
          LENX = N
          LENY = M
      ELSE
          LENX = M
          LENY = N
      END IF
      IF (INCX.GT.0) THEN
          KX = 1
      ELSE
          KX = 1 - (LENX-1)*INCX
      END IF
      IF (INCY.GT.0) THEN
          KY = 1
      ELSE
          KY = 1 - (LENY-1)*INCY
      END IF
! 
! Start the operations. In this version the elements of A are
! accessed sequentially with one pass through A.
! 
! First form  y := beta*y.
! 
      IF (BETA.NE.ONE) THEN
          IF (INCY.EQ.1) THEN
              IF (BETA.EQ.ZERO) THEN
                  DO 10 I = 1,LENY
                      Y(I) = ZERO
   10             CONTINUE
              ELSE
                  DO 20 I = 1,LENY
                      Y(I) = BETA*Y(I)
   20             CONTINUE
              END IF
          ELSE
              IY = KY
              IF (BETA.EQ.ZERO) THEN
                  DO 30 I = 1,LENY
                      Y(IY) = ZERO
                      IY = IY + INCY
   30             CONTINUE
              ELSE
                  DO 40 I = 1,LENY
                      Y(IY) = BETA*Y(IY)
                      IY = IY + INCY
   40             CONTINUE
              END IF
          END IF
      END IF
      IF (ALPHA.EQ.ZERO) RETURN
      IF (LSAME(TRANS,'N')) THEN
! 
! Form  y := alpha*A*x + y.
! 
          JX = KX
          IF (INCY.EQ.1) THEN
              DO 60 J = 1,N
                  IF (X(JX).NE.ZERO) THEN
                      TEMP = ALPHA*X(JX)
                      DO 50 I = 1,M
                          Y(I) = Y(I) + TEMP*A(I,J)
   50                 CONTINUE
                  END IF
                  JX = JX + INCX
   60         CONTINUE
          ELSE
              DO 80 J = 1,N
                  IF (X(JX).NE.ZERO) THEN
                      TEMP = ALPHA*X(JX)
                      IY = KY
                      DO 70 I = 1,M
                          Y(IY) = Y(IY) + TEMP*A(I,J)
                          IY = IY + INCY
   70                 CONTINUE
                  END IF
                  JX = JX + INCX
   80         CONTINUE
          END IF
      ELSE
! 
! Form  y := alpha*A'*x + y.
! 
          JY = KY
          IF (INCX.EQ.1) THEN
              DO 100 J = 1,N
                  TEMP = ZERO
                  DO 90 I = 1,M
                      TEMP = TEMP + A(I,J)*X(I)
   90             CONTINUE
                  Y(JY) = Y(JY) + ALPHA*TEMP
                  JY = JY + INCY
  100         CONTINUE
          ELSE
              DO 120 J = 1,N
                  TEMP = ZERO
                  IX = KX
                  DO 110 I = 1,M
                      TEMP = TEMP + A(I,J)*X(IX)
                      IX = IX + INCX
  110             CONTINUE
                  Y(JY) = Y(JY) + ALPHA*TEMP
                  JY = JY + INCY
  120         CONTINUE
          END IF
      END IF
! 
      RETURN
! 
! End of DGEMV .
! 
      END





