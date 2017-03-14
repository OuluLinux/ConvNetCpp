#include "ConvNet.h"

namespace ConvNet {

LrnLayer::LrnLayer() {
	
}

Volume& LrnLayer::Forward(Volume& input, bool is_training) {
	return output_activation;
}

void LrnLayer::Backward() {
	
}

void LrnLayer::Init(int input_width, int input_height, int input_depth) {
	
}

void LrnLayer::Store(ValueMap& map) const {
	
}

void LrnLayer::Load(const ValueMap& map) {
	
}

String LrnLayer::ToString() const {
	return Format("LRN: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}





#if 0
  // a bit experimental layer for now. I think it works but I'm not 100%
  // the gradient check is a bit funky. I'll look into this a bit later.
  // Local Response Normalization in window, along depths of volumes
  var LrnLayer(opt) {
    var opt = opt || {};

    // required
    k = opt.k;
    n = opt.n;
    alpha = opt.alpha;
    beta = opt.beta;

    // computed
    out_sx = opt.in_sx;
    out_sy = opt.in_sy;
    out_depth = opt.in_depth;
    layer_type = 'lrn';

    // checks
    if(n%2 == 0) { console.log('WARNING n should be odd for LRN layer'); }
  }
  LrnLayer.prototype = {
    forward: function(V, is_training) {
      in_act = V;

      var A = V.cloneAndZero();
      S_cache_ = V.cloneAndZero();
      var n2 = Math.floor(n/2);
      for(var x=0;x<V.sx;x++) {
        for(var y=0;y<V.sy;y++) {
          for(var i=0;i<V.depth;i++) {

            var ai = V.get(x,y,i);

            // normalize in a window of size n
            var den = 0.0;
            for(var j=Math.max(0,i-n2);j<=Math.min(i+n2,V.depth-1);j++) {
              var aa = V.get(x,y,j);
              den += aa*aa;
            }
            den *= alpha / n;
            den += k;
            S_cache_.set(x,y,i,den); // will be useful for backprop
            den = Math.pow(den, beta);
            A.set(x,y,i,ai/den);
          }
        }
      }

      out_act = A;
      return out_act; // dummy identity function for now
    },
    backward: function() { 
      // evaluate gradient wrt data
      var V = in_act; // we need to set dw of this
      V.dw = global.zeros(V.w.length); // zero out gradient wrt data
      var A = out_act; // computed in forward pass 

      var n2 = Math.floor(n/2);
      for(var x=0;x<V.sx;x++) {
        for(var y=0;y<V.sy;y++) {
          for(var i=0;i<V.depth;i++) {

            var chain_grad = out_act.get_grad(x,y,i);
            var S = S_cache_.get(x,y,i);
            var SB = Math.pow(S, beta);
            var SB2 = SB*SB;

            // normalize in a window of size n
            for(var j=Math.max(0,i-n2);j<=Math.min(i+n2,V.depth-1);j++) {
              var aj = V.get(x,y,j); 
              var g = -aj*beta*Math.pow(S,beta-1)*alpha/n*2*aj;
              if(j==i) g+= SB;
              g /= SB2;
              g *= chain_grad;
              V.add_grad(x,y,j,g);
            }

          }
        }
      }
    },
    getParamsAndGrads: function() { return []; },
    toJSON: function() {
      var json = {};
      json.k = k;
      json.n = n;
      json.alpha = alpha; // normalize by size
      json.beta = beta;
      json.out_sx = out_sx;
      json.out_sy = out_sy;
      json.out_depth = out_depth;
      json.layer_type = layer_type;
      return json;
    },
    fromJSON: function(json) {
      k = json.k;
      n = json.n;
      alpha = json.alpha; // normalize by size
      beta = json.beta;
      out_sx = json.out_sx;
      out_sy = json.out_sy;
      out_depth = json.out_depth;
      layer_type = json.layer_type;
    }
  }
  
#endif
