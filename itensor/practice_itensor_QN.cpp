#include<iostream> 
#include<iomanip>
#include<random> 
//#include<string> 
#include <cmath>
#include<eigen3/Eigen/Dense>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor; 

void test_QN()
{
    auto I = itensor::Index(QN(0),1, 
                            QN(1),1, 
                            Out, "I");

    Print(I); 

    auto J = itensor::Index(QN(-2),2, 
                            QN(-1),4, 
                            QN(0), 6, 
                            QN(+1),4,
                            QN(+2),2,
                            Out, "J");
    Print(J); 

    Print(itensor::hasQNs(J));

    
}


void writing_and_reading_from_file()
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

    string filename = "2d_spin_lat_write2file";
    itensor::writeToFile(filename, A); 

    auto A1 = itensor::readFromFile<ITensor>("2d_spin_lat_write2file"); 

    Print(A1); 
}

int main()
{
    //test_QN(); 
    writing_and_reading_from_file();
    return 0 ; 
}