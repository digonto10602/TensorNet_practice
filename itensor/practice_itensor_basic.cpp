#include<iostream> 
#include<iomanip>
#include<random> 
//#include<string> 
#include <cmath>
#include<eigen3/Eigen/Dense>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor; 


void test_basic_functions()
{
    //Definitions 
    auto i = itensor::Index(3,"i ind"); 
    auto j = itensor::Index(3,"j ind"); 
    auto k = itensor::Index(3,"k ind"); 

    auto A = itensor::ITensor(i,j,k); 

    
    
    A.set(i=0, j=0, k=0, 1.56); //indices start from 1 
    A.set(i=1, j=2, k=2, 1.51); 
    A.set(i=1, j=1, k=1, 2.02); 
    A.set(i=4, j=4, k=9, 2.1111); //doesn't do anything 
    
    
    std::cout<<"A = "<<A<<std::endl; 
    std::cout<<"Dim of i "<<i<<std::endl; 
    itensor::println("Dim of i = ",itensor::dim(i)); 
    std::cout<<std::setprecision(20); 
    itensor::println("Order A = ",itensor::order(A)); 

    itensor::println("A = ",A); 
    
    auto x = itensor::elt(A, i=1, j=1, k=1); 
    Print(x); 

    PrintData(A); 
    auto B = 2.0*A; 

    PrintData(B); 

    



}

void test_contraction()
{
    auto d = 3; //index size 
    auto i = itensor::Index(d, "i");
    auto j = itensor::Index(d, "j"); 
    auto k = itensor::Index(d, "k"); 
    auto A = itensor::ITensor(i,j); 
    auto B = itensor::ITensor(j,k); 

    //auto range_of_d = itensor::range1(i); 
    //Print(range_of_d); 

    //compare tensor contraction with Eigen lib:
    Eigen::MatrixXd Amat(d,d);
    Eigen::MatrixXd Bmat(d,d); 

    for(auto i1: itensor::range1(d))
    {
        for(auto j1: itensor::range1(d))
        {
            for(auto k1: itensor::range1(d))
            {
                itensor::println("i,j,k = ",i1,j1,k1); 
                auto value1 = i1*j1*k1; 
                auto value2 = i1*j1+k1; 
                A.set(i=i1,j=j1, value1);
                B.set(j=j1,k=k1, value2);

                Amat(i1-1, j1-1) = value1; 
                Bmat(j1-1, k1-1) = value2; 
            }
        }
    }

    PrintData(A); 
    PrintData(B); 
    auto C = A*B; 
    Eigen::MatrixXd Cmat = Amat*Bmat; 
    PrintData(C); 
    std::cout<<"Cmat = "<<Cmat<<std::endl; 

    auto Aprime = itensor::prime(A,j); 
    auto Asq = A*Aprime; 

    PrintData(Asq); 

}
void factorizing_using_SVD()
{
    auto i = itensor::Index(2,"i ind"); 
    auto j = itensor::Index(5,"j ind"); 
    auto k = itensor::Index(3,"k ind"); 

    auto T = itensor::randomITensor(i,j,k); 

    PrintData(T); 

    auto [U,S,V] = itensor::svd(T,{i,k});

    PrintData(U); 
    PrintData(S);
    PrintData(V);

    auto [U1,S1,V1] = itensor::svd(T,{i,k},{"Cutoff=",1E-2,"MaxDim=",3});
    
    PrintData(U1);
    PrintData(S1);
    PrintData(V1); 

    auto truncerr = itensor::sqr(itensor::norm(U1*S1*V1 - T)/itensor::norm(T));

    Print(truncerr);

}

void TRG_exact_spinsystem(  double T, double &val )
{
    auto pi = std::acos(-1.0); 
    auto firstval = 0.5*std::log(2.0);
    auto secondval = 1.0/(2.0*pi); 

    auto summ = 0.0; 

    auto theta_initial = 0.0; 
    auto theta_final = pi; 
    auto summ_points = 10000.0; 
    auto del_theta = std::abs(theta_initial - theta_final)/summ_points; 

    for(int i=0; i<(int)summ_points + 1; ++i)
    {
        auto theta = theta_initial + ((double)i)*del_theta; 

        auto A = std::pow(std::cosh(2.0/T),2.0); 
        auto kappa = 1.0/std::pow(std::sinh(2/T), 2.0); 
        auto B = (1.0/kappa)*std::sqrt(1 + kappa*kappa - 2.0*kappa*std::cos(2.0*theta)); 

        summ = summ + del_theta * std::log(A + B); 
    }

    //summ = summ/(2.0*pi); 

    val = firstval + secondval*summ; 
}

void TRG_algorithm()
{
    itensor::Real T = 3.0; 
    int maxdim = 20; 
    int topscale = 8; 

    auto dim0 = 2; 
    auto s = itensor::Index(dim0, "scale=0"); 
    auto l = itensor::addTags(s, "left"); 
    auto r = itensor::addTags(s, "right"); 
    auto u = itensor::addTags(s, "up"); 
    auto d = itensor::addTags(s, "down"); 

    auto A = itensor::ITensor(l,r,u,d); 
    auto Sig = [](int s) { return 1.-2.*(s-1); }; 

    for(auto sl : itensor::range1(dim0))
    for(auto sd : itensor::range1(dim0))
    for(auto sr : itensor::range1(dim0))
    for(auto su : itensor::range1(dim0))
    {
        auto E =    Sig(sl)*Sig(sd) + Sig(sd)*Sig(sr)
                 +  Sig(sr)*Sig(su) + Sig(su)*Sig(sl); 
        
        auto P = std::exp(-E/T); 

        A.set(l=sl, d=sd, r=sr, u=su, P); 

        itensor::printfln("\n l=%d, d=%d, r=%d, u=%d, P=%d", sl, sd, sr, su, P);
    }

    itensor::Real z = 1.0; 

    for( auto scale : itensor::range1(topscale))
    {
        itensor::printfln("\n---------Scale %d -> %d ----------", scale, scale+1); 

        auto [Fl, Fr] = itensor::factor(A, {r,d}, {l,u}, {  "MaxDim=",maxdim, 
                                                            "Tags=","left,scale="+itensor::str(scale),
                                                            "ShowEigs=",true }  );
        
        auto l_new = itensor::commonIndex(Fl, Fr);

        auto [Fu, Fd] = itensor::factor(A, {l,d}, {u,r}, {  "MaxDim=",maxdim, 
                                                            "Tags=","up,scale="+itensor::str(scale),
                                                            "ShowEigs=",true }  );

        auto u_new = itensor::commonIndex(Fu, Fd);
        //PrintData(Fl); 
        //PrintData(Fr);
        //PrintData(Fu);
        //PrintData(Fd); 


        //itensor::printfln("\n scale = ",scale); 
               
        //if(scale==5) std::abort(); 

        
        auto r_new = itensor::replaceTags(l_new, "left", "right"); 
        Fr *= itensor::delta(l_new, r_new); 

         
        auto d_new = itensor::replaceTags(u_new, "up", "down"); 
        Fd *= itensor::delta(u_new, d_new); 

        Fl *= itensor::delta(r, l); 

        Fu *= itensor::delta(d, u); //changed this

        Fr *= itensor::delta(l, r); 
         
        Fd *= itensor::delta(u, d); 

        A = Fl * Fu * Fr * Fd; 

        l = l_new; 
        r = r_new; 
        u = u_new; 
        d = d_new; 

        itensor::Real TrA = itensor::elt(A*itensor::delta(l,r)*itensor::delta(u,d)); 

        A /= TrA; 

        z *= std::pow(TrA, 1./std::pow(2, 1+scale)); 
                            
    }

    itensor::printfln("\n ln(Z)/N_s TRG = ", std::log(z)); 

    double exact_z;

    TRG_exact_spinsystem( T, exact_z );

    itensor::printfln("\n ln(Z)/N_s exact = ",exact_z); 
    itensor::printfln("\n diff % = ",std::abs((exact_z - std::log(z))/exact_z)*100.0); 



    
    
}

void TRG_check()
{
    
Real T = 3.0;
int maxdim = 20;
int topscale = 8;

auto dim0 = 2;

// Define an initial Index making up
// the Ising partition function
auto s = Index(dim0,"scale=0");

// Define the indices of the scale-0
// Boltzmann weight tensor "A"
auto l = addTags(s,"left");
auto r = addTags(s,"right");
auto u = addTags(s,"up");
auto d = addTags(s,"down");

auto A = ITensor(l,r,u,d);

// Fill the A tensor with correct Boltzmann weights:
auto Sig = [](int s) { return 1.-2.*(s-1); };
for(auto sl : range1(dim0))
for(auto sd : range1(dim0))
for(auto sr : range1(dim0))
for(auto su : range1(dim0))
    {
    auto E = Sig(sl)*Sig(sd)+Sig(sd)*Sig(sr)
            +Sig(sr)*Sig(su)+Sig(su)*Sig(sl);
    auto P = exp(-E/T);
    A.set(l(sl),r(sr),u(su),d(sd),P);
    }

// Keep track of partition function per site, z = Z^(1/N)
Real z = 1.0;

for(auto scale : range1(topscale))
    {
    printfln("\n---------- Scale %d -> %d  ----------",scale-1,scale);

    // Get the upper-left and lower-right tensors
    auto [Fl,Fr] = factor(A,{r,d},{l,u},{"MaxDim=",maxdim,
                                         "Tags=","left,scale="+str(scale),
                                         "ShowEigs=",true});

    // Grab the new left Index
    auto l_new = commonIndex(Fl,Fr);

    // Get the upper-right and lower-left tensors
    auto [Fu,Fd] = factor(A,{l,d},{u,r},{"MaxDim=",maxdim,
                                         "Tags=","up,scale="+str(scale),
                                         "ShowEigs=",true});

    // Grab the new up Index
    auto u_new = commonIndex(Fu,Fd);

    // Make the new index of Fl distinct
    // from the new index of Fr by changing
    // the tag from "left" to "right"
    auto r_new = replaceTags(l_new,"left","right");
    Fr *= delta(l_new,r_new);

    // Make the new index of Fd distinct
    // from the new index of Fu by changing the tag
    // from "up" to "down"
    auto d_new = replaceTags(u_new,"up","down");
    Fd *= delta(u_new,d_new);

    Fl *= delta(r,l);
    Fu *= delta(d,u);
    Fr *= delta(l,r);
    Fd *= delta(u,d);
    A = Fl * Fu * Fr * Fd;
    
    Print(A);

    // Update the indices
    l = l_new;
    r = r_new;
    u = u_new;
    d = d_new;

    // Normalize the current tensor and keep track of
    // the total normalization
    Real TrA = elt(A*delta(l,r)*delta(u,d));
    A /= TrA;
    z *= pow(TrA,1./pow(2,1+scale));

    }

printfln("log(Z)/N = %.12f",log(z));




}


int main()
{
    //test_basic_functions(); 
    //test_contraction();
    TRG_algorithm();
    //TRG_check(); 
    return 0 ; 
}