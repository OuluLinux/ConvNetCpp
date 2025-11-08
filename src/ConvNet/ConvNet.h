#ifndef _ConvNet_ConvNet_h_
#define _ConvNet_ConvNet_h_

/*

	ConvNet is a neural network library translated from javascript and python.
	
	Original author:
		Andrej Karpathy (C) MIT License
		ConvNetJS Original source:	https://github.com/karpathy/convnetjs
		Reinforce.js sources:		https://github.com/karpathy/reinforcejs
	
	
	
	About translation
	-----------------
	
	Everything has been translated as directly as possible, but because of differences between
	JavaScript and C++ memory model, many functions had to be structurally modified, so they
	may look very different in C++.
	
	The motivation behind this work was to combine the readibility of ConvNetJS and the speed
	of C++. Examples in the ConvNetJS are better than in other NN libraries, but translating
	only them without core library (and using some different NN backend) seemed too difficult.
	The translating felt too prone to errors in the beginning, but after translating
	the first example, I was convinced that it could be done correctly.
	
	I felt much wiser after finishing the translation of all 9 examples of ConvNetJS.
	C++ classes made reading more clear and the practical usage was finally obvious for me.
	I think this was definitely worth of my time and I hope that other find this as
	rewarding as I found.
	
	One usual problem is that arrays are being returned in javascript, but in C++, that
	means slow memory duplications. To fix that, instead of using the return value, the destination
	variable must be in call arguments as a reference, and the function must write directly into
	that. This usually separates C++ version from JS version a lot.
	
	The most difficult problem is JS lambda functions and using a local variable in them, which
	doesn't belong in the function. C++ lambda is not a direct translation, even if it could easily
	seem like that. The recurrent.js / reinforce.js had a callback queue for backpropagation,
	which used those, and it required modular restructuring to work even in the first pass.
	
	The callback queue for backpropagation in RecurrentJS was finally replaced with
	pre-initialized structure. Because of that, no heap allocations are required during forward
	and backward propagation. 
	
	
	Performance
	-----------
	Every heap allocation during training has been avoided by using temporary volumes in
	classes. They are using SetCount(0) in vectors, which does not unallocate the reserved
	memory.
	
	To tune the code even faster, set breakpoints to MemoryAlloc during busy loop, and change
	the code to avoid those by using temporary class variables and by avoiding unallocation
	before destructor. These classes are intended to be used from a single thread, so temp class
	variables shouldn't be a problem.
	
	Multi- or many-thread usage won't be supported in the main repository of ConvNetC++, because
	of the readibility problems.
	
	
	TODO
	----
	 - Forward calculation might be useless in TrainerBase::Train, because it only sets the
	   volume. So maybe use some "SetupForward" instead.
*/

#include "Utilities.h"
#include "Net.h"
#include "LayerBase.h"
#include "Training.h"
#include "Session.h"
#include "MetaSession.h"
#include "Brain.h"
#include "Agent.h"
#include "Recurrent.h"
#include "RecurrentSession.h"
#include "MemoryPool.h"
#include "RuntimeFlexibility.h"
#include "MagicNet.h"

#endif
