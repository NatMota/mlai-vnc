#pragma once
#include <cmath>
#include <stdexcept>
#include <Eigen/Core>

// Use the same OpenBLAS/LAPACK operations as the reference NumPy controller.
// Reordering a reduction by one ULP can change a later contact/parking event.
extern "C" {
double cblas_ddot(int,const double*,int,const double*,int);
void cblas_dsyrk(int,int,int,int,int,double,const double*,int,double,double*,int);
void cblas_dgemv(int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
void dgesdd_(const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*,int*);
void dgesv_(const int*,const int*,double*,const int*,int*,double*,const int*,int*);
}
template<class Derived> double numpy_norm(const Eigen::MatrixBase<Derived>& input){
    auto v=input.eval();return std::sqrt(cblas_ddot(static_cast<int>(v.size()),v.data(),1,v.data(),1));
}
inline double numpy_sigma_min(const Eigen::Matrix<double,3,7>& jac){
    auto a=jac;const char job='N';const int m=3,n=7,lda=3,one=1,lwork=1024;
    double s[3],u=0,vt=0,work[lwork];int iwork[24],info=0;
    dgesdd_(&job,&m,&n,a.data(),&lda,s,&u,&one,&vt,&one,work,&lwork,iwork,&info);
    if(info)throw std::runtime_error("LAPACK SVD failed");return s[2];
}
inline Eigen::Matrix<double,7,1> numpy_ik(const Eigen::Matrix<double,3,7>& jac,const std::array<bool,7>& free,double damping,const Eigen::Vector3d& velocity){
    Eigen::Matrix<double,3,7> compact;int indices[7],n=0;
    for(int j=0;j<7;j++)if(free[j]){indices[n]=j;compact.col(n++)=jac.col(j);}
    Eigen::Matrix<double,7,1> result=Eigen::Matrix<double,7,1>::Zero();if(!n)return result;
    // NumPy J[:, free] is Fortran-contiguous; matmul uses row-major SYRK,
    // Trans, Upper, then copies that triangle to make the symmetric product.
    Eigen::Matrix<double,3,3,Eigen::RowMajor> gram;
    cblas_dsyrk(101,121,112,3,n,1.,compact.data(),3,0.,gram.data(),3);
    for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)gram(j,i)=gram(i,j);
    for(int i=0;i<3;i++)gram(i,i)+=damping*damping;
    Eigen::Matrix3d solve_matrix=gram;Eigen::Vector3d solution=velocity;
    const int three=3,one=1;int pivots[3],info=0;
    dgesv_(&three,&one,solve_matrix.data(),&three,pivots,solution.data(),&three,&info);
    if(info)throw std::runtime_error("LAPACK IK solve failed");
    double outputs[7];
    cblas_dgemv(102,112,3,n,1.,compact.data(),3,solution.data(),1,0.,outputs,1);
    for(int i=0;i<n;i++)result[indices[i]]=outputs[i];return result;
}
