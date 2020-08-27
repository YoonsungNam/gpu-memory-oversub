typedef struct {
     void *ptr ; //address of the data 
     size_t size; //number of bytes in this parameter 
     volatile int *done_flag; // Indicates readiness of data
     unsigned char type; //Indicates input and/or output from device 
 } param;

typedef struct task {
     int type ; // what code to run for this task 
     param *p; // list of parameters to this task
     int *ready_flag; //indicates if this task has completed
     itn num_params; //number of pararmetners
} task;



