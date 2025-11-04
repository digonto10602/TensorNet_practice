#include<iostream> 
#include<iomanip>
#include<random> 
//#include<string> 
#include <cmath>
#include<eigen3/Eigen/Dense>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
//#include "itensor/mpi.h"
#include "itensor/util/parallel.h"
#include "mpi.h"

void first_MPI_program_test(itensor::Environment &env, int argc, char* argv[])
{
    //do this once in the main file, if this end once
    //will throw error 
    //itensor::Environment env(argc, argv); 

    if(env.firstNode())
    {
        itensor::printfln("There are %d nodes", env.nnodes()); 

    }
}

void simple_communication(itensor::Environment &env, int argc, char* argv[])
{
    //do it once in the main file 
    //itensor::Environment env(argc, argv); 

    int i = 0;
    if(env.firstNode())
    {
        i = 5;
    }

    
    itensor::printfln("node %d has i=%d", env.rank(), i); 

    itensor::broadcast(env, i); 

    itensor::printfln("Now node %d has i=%d", env.rank(), i); 
}

void mailbox_examle(itensor::Environment &env, int argc, char* argv[])
{
    if(env.rank() == 0)
    {

        double actual_result = 1.0; 

        for(int i=1; i<10; ++i)
        {
            actual_result = actual_result * (double) i; 
        }
        for(int i=0; i<10; ++i)
        {
            actual_result = actual_result + (double) i;
        }
        itensor::printfln("1.Actual Result = %d", actual_result);

        double result = 1.0; 

        for(int i=1; i<10; ++i)
        {
            result = result * (double) i; 
        }

        itensor::printfln("2.Node = %d , multiplication result = %d", env.rank(), result); 

        double sum_result = 0.0;
        //setup mailbox to send to node 1;
        auto mbox = itensor::MailBox(env, 1); 
        mbox.send(result); 
        itensor::printfln("3.Node %d to Node %d, message sent. result = %d", env.rank(), env.rank()+1, result);

        auto mbox_recv = itensor::MailBox(env, 1); 
        mbox_recv.receive(sum_result);
        MPI_Status status;
        double sum_result_new_message = 0.0; 
        MPI_Recv(&sum_result_new_message, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);


        // status.MPI_SOURCE tells you which rank sent the message
        int sender_rank = status.MPI_SOURCE;
        
         
        itensor::printfln("8.Node %d to Node %d, message received, result = %d", sender_rank, env.rank(), sum_result );
        itensor::printfln("9.Check value of new message %d", sum_result_new_message); 
        itensor::printfln("10.Delta = %d", std::abs(actual_result - sum_result)); 
    }
    else if(env.rank() == 1)
    {
        auto mbox = itensor::MailBox(env, 0);

        double received_value = 0.0; 
        mbox.receive(received_value); 

        itensor::printfln("4.Node 1 received value = %d", received_value); 

        auto summed_result = received_value; 

        for(int i=0; i<10; ++i)
        {
            summed_result = summed_result + (double) i; 
        }


        itensor::printfln("5.Node %d , summed result = %d", env.rank(), summed_result); 

        auto mbox_send = itensor::MailBox(env, 0); 
        mbox_send.send(summed_result); 
        itensor::printfln("6.Node 1 message sent to Node 0, val = %d",summed_result); 
        
        //mbox_send.send(summed_result);
        MPI_Send(&summed_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        itensor::printfln("7.Node 1 message sent to Node 0 again, val = %d",summed_result);
    }
}



int main(int argc, char* argv[])
{
    //This can only be invoked once 
    itensor::Environment env1(argc, argv);
    //itensor::Environment env2(argc, argv);
    //first_MPI_program_test(env1, argc, argv);
    //simple_communication(env1, argc, argv);
    mailbox_examle(env1, argc, argv);
    return 0; 
}