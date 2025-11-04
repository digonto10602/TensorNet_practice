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

void LatticeBond_example()
{
    int Nx = 3; 
    int Ny = 3; 

    auto Lattice = itensor::squareLattice(Nx, Ny); 

    for(auto &bond : Lattice)
    {
        itensor::printfln("Bond from site %d -> %d", bond.s1, bond.s2); 
        itensor::printfln(" Connecting points (%s,%s) -> (%s,%s)", bond.x1, bond.y1, bond.x2, bond.y2);
        itensor::printfln(" This bond of type \"%s\"", bond.type); 
    }

}

void different_lattices()
{
    int Nx = 3; 
    int Ny = 3; 

    auto square_latt = itensor::squareLattice(Nx, Ny); 
    auto square_latt_nn = itensor::squareNextNeighbor(Nx, Ny, {"YPeriodic=",true});
    auto tri_latt = itensor::triangularLattice(Nx, Ny);

    //printing lattice 
    itensor::printfln("Square lattice"); 
    for(auto &bond : square_latt)
    {
        itensor::printfln("Bond from %d -> %d", bond.s1, bond.s2);
    }
    itensor::printfln("SquareNN lattice");
    for(auto &bond : square_latt_nn)
    {
        if(bond.type == "1")
        {
            itensor::printfln("1st neighbor bond from %d -> %d", bond.s1, bond.s2);
        }
        else if(bond.type == "2")
        {
            itensor::printfln("2nd neighbor bond from %d -> %d", bond.s1, bond.s2); 
        }
    }
    itensor::printfln("Triangular Lattice");
    for(auto &bond : tri_latt)
    {
        itensor::printfln("Bond from %d -> %d", bond.s1, bond.s2); 
        //std::cout<<bond.x1<<std::endl; 
        itensor::printfln(" Connecting points (%s,%s) -> (%s,%s)", bond.x1, bond.y1, bond.x2, bond.y2);
    }
    
    //AutoMPO 
    int N = Nx*Ny; 
    auto sites = itensor::SpinHalf(N); 
    auto ampo = itensor::AutoMPO(sites); 

    for(auto &bond : square_latt)
    {
        ampo += "Sz", bond.s1, "Sz", bond.s2;
        ampo += 0.5, "S+", bond.s1, "S-", bond.s2; 
        ampo += 0.5, "S-", bond.s1, "S+", bond.s2; 
    }

    auto H = itensor::toMPO(ampo); 

    std::cout<<H<<std::endl; 

}

int main( int argc, char* argv[])
{
    LatticeBond_example();
    different_lattices();

    return 0; 
}