#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include "stdlib.h"
#include "time.h"

#include "dendro.h"
#include "graph.h"
#include "rbtree.h"

using namespace std;

// ******** Function Prototypes ***************************************************************************

bool		markovChainMonteCarlo();
bool		parseCommandLine(int argc, char * argv[]);


// ******** Structures and Constants **********************************************************************

struct ioparameters {
	int			n;				// number vertices in input graph
	int			m;				// number of edges in input graph

	string		d_dir;			// working directory
	string		f_in;			// name of input file (either .pairs or .hrg)
	bool			flag_f;			// flag for if -f invoked
	string		f_dg;			// name of output hrg file
	string		f_dg_info;		// name of output information-on-hrg file
	string		f_stat;			// name of output statistics file
	string		f_pairs;			// name of output random graph file
	string		f_namesLUT;		// name of output names LUT file
	bool			flag_make;		// flag for if -make invoked
	string		s_scratch;		// filename sans extension
	string		s_tag;			// user defined filename tag
	string		start_time;		// time simulation was started
	int			timer;			// timer for reading input
	bool			flag_timer;		// flag for when timer fires
	bool			flag_compact;		// compact the Lxy file
};

// ******** Global Variables ******************************************************************************

ioparameters	ioparm;				// program parameters
rbtree		namesLUT;				// look-up table; translates input file node names to graph indices
dendro*		d;					// hrg data structure
unsigned int	t;					// number of time steps max = 2^32 ~ 4,000,000,000
double		bestL;				// best likelihood found so far
int			out_count;			// counts number of maximum found
unsigned int	period  = 10000;		// number of MCMC moves to do before writing stuff out; default: 10000
double*		Likeli;				// holds last k hrg likelihoods

// ******** Main Loop *************************************************************************************

int main(int argc, char * argv[]) {
    int nNode = 0;
	for(int i = 0; i < 16; i++){
        if (namesLUT.findItem(i) == NULL) { namesLUT.insertItem(i, nNode++);}
    }
    d->g = new graph (nNode);

}

