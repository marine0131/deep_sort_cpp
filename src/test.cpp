#include <Eigen/Dense>
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    // Eigen::Matrix4d projected_cov;
    // projected_cov << 
    //     1.44,0,0,1,
    //     0,1.44,0,0,
    //     0,0,1.038,0,
    //     0,0,0,1.44;
    // Eigen::MatrixXd a = projected_cov.rowwise().normalized();
    // cout << 1.0 -a.array() <<endl;
    //
    //
    vector<float> a = {1,2,3,4,5,6,7};
    for(size_t i = 0; i<2;++i)
    {
        a.erase(a.begin());
    }

    for(vector<float>::iterator it=a.begin(); it!=a.end(); ++it)
        cout << *it <<endl;

    // Eigen::LLT<Eigen::Matrix4d> cholesky_mat;                                   
    // Eigen::Matrix4d projected_cov_sqrt = projected_cov.array().sqrt();
    // cholesky_mat = projected_cov_sqrt.llt();                                         
    // Eigen::MatrixXd z;                                                          
    // z.resize(4, 1);                                                             
    // z << 
    //     5,6,7,2;

    // Eigen::MatrixXd mahalanobis = cholesky_mat.solve(z); 
    // cout << mahalanobis <<endl;
}
