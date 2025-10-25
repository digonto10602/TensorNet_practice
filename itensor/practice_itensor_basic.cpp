#include<iostream> 
#include<iomanip>
#include "itensor/all.h"


void test_basic_functions()
{
    
    auto i = itensor::Index(3,"i ind"); 
    auto j = itensor::Index(3,"j ind"); 
    auto k = itensor::Index(3,"k ind"); 

    auto A = itensor::ITensor(i,j,k); 

    
    
    A.set(i=1, j=2, k=2, 1.51); 
    A.set(i=1, j=1, k=1, 2.02); 

    
    
    std::cout<<"A = "<<A<<std::endl; 
    std::cout<<"Dim of i "<<i<<std::endl; 
    itensor::println("Dim of i = ",itensor::dim(i)); 
    std::cout<<std::setprecision(20); 
    itensor::println("Order A = ",itensor::order(A)); 

    itensor::println("A = ",A); 

}

int main()
{
    test_basic_functions(); 

    return 0 ; 
}