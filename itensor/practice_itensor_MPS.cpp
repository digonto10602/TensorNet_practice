#include<iostream> 
#include<iomanip>
#include<random> 
//#include<string> 
#include <cmath>
#include<eigen3/Eigen/Dense>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
//#include "itensor/mpi.h"
//#include "itensor/util/parallel.h"
//#include "mpi.h"



void MPS_basic_example()
{
    //practice sites

    auto N = 10;
    //auto sites = itensor::SpinHalf(N);

    auto sites = itensor::Electron(N);

    Print(itensor::length(sites)); 
    auto s3 = sites(3); 
    Print(s3); 
    auto s3down = sites(3, "Dn"); 
    auto s3up = sites(3, "Up");
    
    Print(s3up);
    Print(s3down); 

    
    //check op method:

    //sites = itensor::Electron(N); 
    auto initstate = itensor::InitState(sites,"Up");
    auto psi = itensor::randomMPS(initstate);

    auto ndn5 = itensor::op(sites,"Ndn",5);

    auto dens = itensor::elt(itensor::dag(itensor::prime(psi(8),"Site")) * itensor::op(sites,"Ntot",8) * psi(8));

    Print(psi); 
    Print(ndn5);
    Print(dens);  
}

void MPS_example()
{
    int N = 10; 
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=",false}); 

    auto A = itensor::randomMPS(sites); 

    A.position(1); 

    auto j = 5; 
    A.position(j); 

    Print(A); 
    auto Aj = A(j); 
    Print(Aj);
    
    A.set(j, 2*Aj); 
    Print(A); 

    A.ref(j) *= -1; 
    Print(A); 

    auto state = itensor::InitState(sites); 
    for(int i=1; i<=N; ++i)
    {
        if(i%2==0) state.set(i, "Up"); 
        else state.set(i, "Dn"); 
    }

    auto B = itensor::MPS(state); 

    Print(B); 

    Print(itensor::isOrtho(A));
    Print(itensor::orthoCenter(A)); 

    //auto expRes = itensor::expect(B, sites, "Sz"); 

    //Print(expRes);

    //auto czz = itensor::correlationMatrix(A,sites,"Sz","Sz");
    //Print(czz);
}

void expectation_value_example()
{
    int N = 10; 
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=",false}); 

    auto A = itensor::randomMPS(sites);
    auto czz = itensor::correlationMatrix(A,sites,"Sz","Sz");
    //Print(czz);
    //itensor::printfln("<Sz_3 Sz_7> = ", czz(1));

    for(int i = 0; i < N; ++i) {
    for(int j = 0; j < N; ++j) {
      itensor::printfln("czz i=%d, j=%d, czz=%d",i,j,czz[i][j]);
    }
    itensor::printfln("");
  }

}

void DMRG_example() 
{
    int N = 100;
    auto sites = itensor::SpinHalf(N, {"ConserveQNs=", false});

    auto ampo = itensor::AutoMPO(sites);
    for(int j = 1; j < N; ++j) {
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
        ampo +=    "Sz",j,"Sz",j+1;
    }
    auto H = itensor::toMPO(ampo);

    auto sweeps = itensor::Sweeps(5);
    sweeps.maxdim() = 10, 20, 100, 100, 200;
    sweeps.cutoff() = 1E-10;

    auto psi0 = itensor::randomMPS(sites);
    auto [energy,psi] = itensor::dmrg(H,psi0,sweeps);
    itensor::printfln("Ground State Energy = ",energy);
    itensor::printfln("psi = ",psi);
}

int main( int argc, char* argv[])
{
    //MPS_example(); 
    //expectation_value_example();

    DMRG_example();

    return 0; 
}