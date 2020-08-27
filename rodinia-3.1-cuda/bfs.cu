/*
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define MAX_THREADS_PER_BLOCK 512
typedef cudaStream_t cudaStream_t;

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
	///
	const int N=100;
	const int nStreams=2;
	const int nBlocks=2;
	const int nThreads=2;
	int offset=0;

	cudaStream_t* streams=(cudaStream_t*)malloc(nStreams*sizeof(cudaStream_t));
	for(int i=0;i<nStream;i++)
		cudaStreamCreate(&(streams[i]));
	////
    char *input_f;
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	/*
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
    */
    //allocated unified memory (UM) -> using only d_*_nodes pointers
    Node* d_graph_nodes;
    cudaMallocManaged(&d_graph_nodes, sizeof(Node)*no_of_nodes);
	
    bool *d_graph_mask;
    cudaMallocManaged(&d_graph_mask, sizeof(bool)*no_of_nodes);
	
    bool *d_updating_graph_mask;
    cudaMallocManaged(&d_updating_graph_mask, sizeof(bool)*no_of_nodes);
	
    bool *d_graph_visited;
    cudaMallocManaged(&d_graph_visited, sizeof(bool)*no_of_nodes);


	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
        // change h_* variables to d_* ones
		d_graph_nodes[i].starting = start;
		d_graph_nodes[i].no_of_edges = edgeno;
		d_graph_mask[i]=false;
		d_updating_graph_mask[i]=false;
		d_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask (unified memory)
	d_graph_mask[source]=true;
	d_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;

	//Copy the Edge List to device Memory
	int* d_graph_edges;
    cudaMallocManaged (&d_graph_edges, sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		d_graph_edges[i] = id;
        //printf("d_graph_edges[%d]: %d\n", i, d_graph_edges[i]);
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");
    /*
	//Copy the Node list to device memory
	Node* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Edge List to device Memory
	int* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;

	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
    
	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
    */
    /*
	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	// allocate device memory for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;
    */
	
    // allocate device memory for result (unified memory)
	int* d_cost;
    cudaMallocManaged (&d_cost, sizeof(int)*no_of_nodes);
    for(int i=0;i<no_of_nodes;i++)
		d_cost[i]=-1;
	d_cost[source]=0;


	//make a bool to check if the execution is over (unified memory)
	bool *d_over;
	cudaMallocManaged(&d_over, sizeof(bool));

	//cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("Using Unified Memory... all cudaMallocManaged finished.\n");
	//printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	//printf("Start traversing the tree\n");
	//bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	//int last_progress=0;
	int* last_progress;
	cudaMallocManaged(&last_progress, sizeof(int)*1);
	last_progress[0] = 0;
	do
	{
        *d_over = false;
		// caching code here
		// caching data here, too
		// cudaMemPrefetchAsync( , , , );
		Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes, last_progress);
		// check if kernel execution generated and error
		
		cudaDeviceSynchronize();

		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
		cudaDeviceSynchronize();
		k++;
        printf("k: %d, d_over: %d\n", k, *d_over);
	}
	while(*d_over);


	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
	//cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,d_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
    /*
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
    */
	for(int i=0;i<nStream;i++)
		cudaStreamDestroy((streams[i]));
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
}
