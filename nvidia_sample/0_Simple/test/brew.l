%{
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>


using namespace std;

typedef struct parameter {
    string dtype;
    string name;
} prm ;

string funct, d_funct, space;
string dim, arg, token, param;
vector <string> dims, args;
vector <prm> params;

%}

%%
.*<<<.*>>>.*\(.*\)[[:blank:]\n]*; {
//host call 
                cout << yytext << endl; 

                funct = yytext;
                dim = yytext;
                arg = yytext;

                //catch call statement
                funct = funct.substr(0, funct.find('<'));

                //catch dimensions
                int start = dim.find_last_of('<') + 1;
                dim = dim.substr(start, dim.find_first_of('>')-start); 
               
                stringstream ss(dim); 
                while(getline(ss, token, ','))
                   dims.push_back(token);

               //catch arguments 
                arg = arg.substr(arg.find('(')+1, arg.find(')')-(arg.find('(')+1));
                stringstream sarg(arg); 
                while(getline(sarg, token, ','))
                    args.push_back(token);
/*
                cout << "   " << funct << "<<< "<< dims.at(0) << "/2" <<" , " << dims.at(1) << "/2"<< " >>> " << " ( ";
                for ( auto iter = args.begin() ; iter != args.end() ; iter++) 
                    if (iter != args.end() - 1) 
                        cout << *iter << "/2" << " , ";
                    else
                        cout << *iter << "/2" << " ) ;       //changed code"<<endl;
*/
            }


.*__.*__.*\(.*\)[[:blank:]\n]*\{     {
//kernel fuction                          

                cout << "#define WARP_SZ 32" << endl;
                cout << "__device__ inline int lane_id (void) { return threadIdx.x % WARP_SZ; }" << endl;
                cout << "static __device__ __inline__ uint64_t get_smid(void) { " << endl;
                cout << "\t\tuint32_t ret;" << endl;
                cout << "\t\tasm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(ret) );" << endl;
                cout << "\t\treturn ret; \n }" << endl << endl;


                cout << yytext << endl;
                cout << "\t\tint th_idx = threadIdx.x+blockDim.x*blockIdx.x;" << endl << endl;
                
                cout << "\t\t__shared__ uint64_t id, valid_task;" << endl;
                cout << "\t\tuint64_t smid;" << endl << endl;

                cout << "// There is 30 SMs in TITAN XP ( Each MP has 128 CUDA Cores, 4 WARPS)" << endl;
                cout << "\t\tuint64_t sm_low = 0;" << endl;
                cout << "\t\tuint64_t sm_high = 14; " << endl;
                cout << "\t\tconst int leader = ( ! threadIdx.x && ! threadIdx.y && ! threadIdx.z ); " << endl << endl;
               
                cout << "\t\tif (leader) {" << endl;
                cout << "\t\t\tid = 0;" << endl;
                cout << "\t\t\tsmid = get_smid();" << endl;
                cout << "\t\t\tvalid_task = (sm_low <= smid && smid <= sm_high); " << endl;
                cout << "\t\t}" << endl << endl;

                cout << "\t\t__syncthreads(); " << endl;
                cout << "\t\tif (!valid_task) {return;} " << endl << endl; 
                
                space = yytext; param = yytext; d_funct = yytext; 
                //catch device space
                int start = space.find('_', 1) + 1;
                space  = space.substr(start, space.find('_', 2) - start);
                
                //catch function name
                d_funct = d_funct.erase(0, space.size());
                start = d_funct.find('_', 3) + 1;
                d_funct = d_funct.substr( start, d_funct.find_first_of('(') - start);

                //catch parameters
                param = param.substr(param.find('(')+1, param.find(')') - (param.find('(')+1));
                stringstream sp(param);


                while(getline(sp, token, ','))  {
                    prm temp;
                    int pos = token.find('*');
                    
                    if (pos == -1) 
                        pos = token.find (' ');
                    
                    temp.dtype  = token.substr( 0 , pos+1 );
                    temp.name  = token.substr(pos + 1, token.size() - (pos + 1));
                    params.push_back(temp);
                }
/*
                cout << "  __" << space << "__" << d_funct <<" ( ";
                
                for ( auto iter = params.begin() ; iter != params.end() ; iter++) 
                    if (iter != params.end() - 1) 
                        cout << (*iter).dtype << "  " << (*iter).name << " , ";
                    else
                        cout << (*iter).dtype << "  " << (*iter).name << " ) {       //changed code"<<endl;
  */              
                } 
%%

int main (int argc, char ** argv) {
   cout << " /* " << endl << "*" <<endl << "* header of generated code " << endl << "* " << endl << "*/ " <<endl;
   cout << "# include <stdint.h>" <<endl;
   int result = yylex();

   cout << endl << "/*"<< endl << "[HOST]" <<endl; 
   cout << "--- function: " << funct << endl;
   cout << "--- dimensions: ";
   for ( auto iter = dims.begin(); iter != dims.end() ; ++ iter) 
       cout << *iter << "\t" ;
   cout<<endl<<"--- arguments:";
   for( auto iter = args.begin(); iter != args.end(); ++iter)
        cout << "\t" << *iter;
   
   //print kernel info
   cout << endl << endl << "[KERNEL]" << endl;
   cout << "--- space: " << space << endl;
   cout << "--- function: " << d_funct << endl;
   cout << "--- arguments:";
   for( auto iter = params.begin(); iter != params.end(); ++iter)
       cout << "\t" << "[ '" << iter->dtype << "' , '" << iter->name << " ']";
   cout << endl << "*/" <<endl;
   return 0;
}

