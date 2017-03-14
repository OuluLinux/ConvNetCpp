#ifndef _GridWorld_GridWorld_h
#define _GridWorld_GridWorld_h

#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace ConvNet;
using namespace Upp;

#define IMAGECLASS GridWorldImg
#define IMAGEFILE <GridWorld/GridWorld.iml>
#include <Draw/iml_header.h>


class GridWorld : public TopWindow {
	
	DPAgent agent;
	Vector<double> Rarr, T;
	
	int gh, gw, gs;
	int selected;
	int sid;
	bool jaxrendered;
	
public:
	typedef GridWorld CLASSNAME;
	GridWorld();
	
	
	void Reset() {
		
		// hardcoding one gridworld for now
		gh = 10;
		gw = 10;
		gs = gh * gw; // number of states
		
		// specify some rewards
		Rarr.SetCount(gs);
		T.SetCount(gs);
		
		Rarr[55] = 1;
		
		Rarr[54] = -1;
		//Rarr[63] = -1;
		Rarr[64] = -1;
		Rarr[65] = -1;
		Rarr[85] = -1;
		Rarr[86] = -1;
		
		Rarr[37] = -1;
		Rarr[33] = -1;
		//Rarr[77] = -1;
		Rarr[67] = -1;
		Rarr[57] = -1;
		
		// make some cliffs
		for (int q = 0; q < 8; q++) {
			int off = (q+1)*gh+2;
			T[off] = 1;
			Rarr[off] = 0;
		}
		for (int q = 0; q < 6; q++) {
			int off = 4*gh+q+2;
			T[off] = 1;
			Rarr[off] = 0;
		}
		
		T[5*gh+2] = 0; Rarr[5*gh+2] = 0; // make a hole
		/*Rarr = Rarr;
		T = T;
		*/
		Panic("chk");
		
		selected = -1;
		sid = -1;
	}
	
	double Reward(int s, int a, int ns) {
		// reward of being in s, taking action a, and ending up in ns
		return Rarr[s];
	}
	
	int NextStateDistribution(int s, int a) {
		int ns;
		// given (s,a) return distribution over s' (in sparse form)
		if (T[s] == 1) {
			// cliff! oh no!
			// var ns = 0; // reset to state zero (start)
			ASSERT(T[s] != 1);
		} else if(s == 55) {
			// agent wins! teleport to start
			ns = StartState();
			while(T[ns] == 1) {
				ns = RandomState();
			}
		} else {
			// ordinary space
			double  nx, ny;
			double  x = stox(s);
			double  y = stoy(s);
			if (a == 0) {nx=x-1; ny=y;}
			if (a == 1) {nx=x; ny=y-1;}
			if (a == 2) {nx=x; ny=y+1;}
			if (a == 3) {nx=x+1; ny=y;}
			ns = nx*gh+ny;
			if(T[ns] == 1) {
				// actually never mind, this is a wall. reset the agent
				ns = s;
			}
		}
		// gridworld is deterministic, so return only a single next state
		return ns;
	}
	
	/*SampleNextState(s, a) {
		// gridworld is deterministic, so this is easy
		int ns = NextStateDistribution(s, a);
		double r = Rarr[s]; // observe the raw reward of being in s, taking a, and ending up in ns
		r -= 0.01; // every step takes a bit of negative reward
		var out = {'ns':ns, 'r':r};
		if (s == 55 && ns == 0) {
			// episode is over
			out.reset_episode = true;
		}
		return out;
	}*/
	
	Vector<int> AllowedActions(double s) {
		double x = stox(s);
		double y = stoy(s);
		Vector<int> as;
		if(x > 0) { as.Add(0); }
		if(y > 0) { as.Add(1); }
		if(y < gh-1) { as.Add(2); }
		if(x < gw-1) { as.Add(3); }
		return as;
	}
	
	int RandomState() { return Random(gs); }
	int StartState() { return 0; }
	int GetNumStates() { return gs; }
	int GetMaxNumActions() { return 4; }
	
	// private functions
	int stox(int s) { return floor(s/gh); }
	int stoy(int s) { return s % gh; }
	int xytos(int x, int y) { return x*gh + y; }
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	// ------
	// UI
	// ------
	/*var rs = {};
	var trs = {};
	var tvs = {};
	var pas = {};
	var cs = 60;  // cell size*/
	
	void InitGrid() {
		/*
		var d3elt = d3.select('#draw');
		d3elt.html('');
		rs = {};
		trs = {};
		tvs = {};
		pas = {};
		
		var gh= env.gh; // height in cells
		var gw = env.gw; // width in cells
		var gs = env.gs; // total number of cells
		
		var w = 600;
		var h = 600;
		svg = d3elt.append('svg').attr('width', w).attr('height', h)
			.append('g').attr('transform', 'scale(1)');
		
		// define a marker for drawing arrowheads
		svg.append("defs").append("marker")
			.attr("id", "arrowhead")
			.attr("refX", 3)
			.attr("refY", 2)
			.attr("markerWidth", 3)
			.attr("markerHeight", 4)
			.attr("orient", "auto")
			.append("path")
			.attr("d", "M 0,0 V 4 L3,2 Z");
		
		for (int y = 0; y < gh; y++) {
			for (int x = 0; x < gw; x++) {
				int xcoord = x*cs;
				int ycoord = y*cs;
				var s = env.xytos(x,y);
				
				var g = svg.append('g');
				// click callbackfor group
				g.on('click', function(ss) {
					return function() { cellClicked(ss); } // close over s
				}(s));
				
				// set up cell rectangles
				var r = g.append('rect')
					.attr('x', xcoord)
					.attr('y', ycoord)
					.attr('height', cs)
					.attr('width', cs)
					.attr('fill', '#FFF')
					.attr('stroke', 'black')
					.attr('stroke-width', 2);
				rs[s] = r;
				
				// reward text
				var tr = g.append('text')
					.attr('x', xcoord + 5)
					.attr('y', ycoord + 55)
					.attr('font-size', 10)
					.text('');
				trs[s] = tr;
				
				// skip rest for cliffs
				if(env.T[s] == 1) { continue; }
				
				// value text
				var tv = g.append('text')
					.attr('x', xcoord + 5)
					.attr('y', ycoord + 20)
					.text('');
				tvs[s] = tv;
				
				// policy arrows
				pas[s] = []
				for(var a=0;a<4;a++) {
					var pa = g.append('line')
						.attr('x1', xcoord)
						.attr('y1', ycoord)
						.attr('x2', xcoord)
						.attr('y2', ycoord)
						.attr('stroke', 'black')
						.attr('stroke-width', '2')
						.attr("marker-end", "url(#arrowhead)");
					pas[s].push(pa);
				}
			}
		}
		*/
	}
	
	void DrawGrid() {
		/*
		int gh = env.gh; // height in cells
		int gw = env.gw; // width in cells
		int gs = env.gs; // total number of cells
		
		int cs = 0;
		
		// updates the grid with current state of world/agent
		for (int y = 0; y < gh; y++) {
			for (int x = 0; x < gw; x++) {
				int xcoord = x*cs;
				int ycoord = y*cs;
				int r=255, g=255, b=255;
				int s = xytos(x, y);
				
				double vv = agent.V[s];
				int ms = 100;
				if (vv > 0) { g = 255; r = 255 - vv*ms; b = 255 - vv*ms; }
				if (vv < 0) { g = 255 + vv*ms; r = 255; b = 255 + vv*ms; }
				Color vcol(r, g, b);
				if (T[s] == 1) { vcol = "#AAA"; rcol = "#AAA"; }
				
				// update colors of rectangles based on value
				var r = rs[s];
				if(s == selected) {
					// highlight selected cell
					r.attr('fill', '#FF0');
				} else {
					r.attr('fill', vcol);
				}
				
				// write reward texts
				double rv = Rarr[s];
				double tr = trs[s];
				if(rv !== 0) {
					tr.text('R ' + rv.toFixed(1))
				}
				
				// skip rest for cliff
				if (T[s] == 1) continue;
				
				// write value
				var tv = tvs[s];
				tv.text(agent.V[s].toFixed(2));
				
				// update policy arrows
				var paa = pas[s];
				for(var a=0;a<4;a++) {
					var pa = paa[a];
					var prob = agent.P[a*gs+s];
					if(prob == 0) { pa.attr('visibility', 'hidden'); }
					else { pa.attr('visibility', 'visible'); }
					var ss = cs/2 * prob * 0.9;
					if (a == 0) {nx=-ss; ny=0;}
					if (a == 1) {nx=0; ny=-ss;}
					if (a == 2) {nx=0; ny=ss;}
					if (a == 3) {nx=ss; ny=0;}
					pa.attr('x1', xcoord+cs/2)
						.attr('y1', ycoord+cs/2)
						.attr('x2', xcoord+cs/2+nx)
						.attr('y2', ycoord+cs/2+ny);
				}
			}
		}
		*/
	}
	
	
	void CellClicked(int s) {
		/*
		if(s == selected) {
			selected = -1; // toggle off
			$("#creward").html('(select a cell)');
		} else {
			selected = s;
			$("#creward").html(Rarr[s].toFixed(2));
			$("#rewardslider").slider('value', Rarr[s]);
		}
		DrawGrid(); // redraw
		*/
	}
	
	void UpdatePolicy() {
		agent.UpdatePolicy();
		DrawGrid();
	}
	
	void EvaluatePolicy() {
		agent.EvaluatePolicy();
		DrawGrid();
	}
	
	
	void runValueIteration() {
		if (sid == -1) {
			/*sid = setInterval(function(){
				agent.evaluatePolicy();
				agent.updatePolicy();
				DrawGrid();
			}, 100);*/
		} else {
			//ClearInterval(sid);
			sid = -1;
		}
	}
	
	void ResetAll() {
		Reset();
		agent.Reset();
		DrawGrid();
	}
	
	
	void Start() {
		//agent = new DPAgent(env, {'gamma':0.9}); // create an agent, yay!
		agent.SetGamma(0.9);
		InitGrid();
		DrawGrid();
		
		/*$("#rewardslider").slider({
			min: -5,
			max: 5.1,
			value: 0,
			step: 0.1,
			slide: function(event, ui) {
				if(selected >= 0) {
					env.Rarr[selected] = ui.value;
					$("#creward").html(ui.value.toFixed(2));
					DrawGrid();
				} else {
					$("#creward").html('(select a cell)');
				}
			}
		});
		
		// suntax highlighting
		//marked.setOptions({highlight:function(code){ return hljs.highlightAuto(code).value; }});
		$(".md").each(function(){
			$(this).html(marked($(this).html()));
		});
		renderJax();*/
	}
	
	/*var jaxrendered = false;
	function renderJax() {
		if(jaxrendered) { return; }
		(function () {
			var script = document.createElement("script");
			script.type = "text/javascript";
			script.src  = "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";
			document.getElementsByTagName("head")[0].appendChild(script);
			jaxrendered = true;
		})();
	}*/
};

#endif
