// Inference for Llama-2 Transformer model in pure go.
package cmd

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/spf13/cobra"
)

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "run",
	Run: func(cmd *cobra.Command, args []string) {
		modelPath, err := cmd.Flags().GetString("model")
		if err != nil {
			log.Fatalln(err)
		}
		tokenizerPath, err := cmd.Flags().GetString("tokenizer")
		if err != nil {
			log.Fatalln(err)
		}
		temperature, err := cmd.Flags().GetFloat32("temperature")
		if err != nil {
			log.Fatalln(err)
		}
		steps, err := cmd.Flags().GetInt("steps")
		if err != nil {
			log.Fatalln(err)
		}

		f, err := os.Open(modelPath)
		if err != nil {
			log.Fatalln(err)
		}
		defer f.Close()

		config, err := NewConfig(f)
		if err != nil {
			log.Fatalln(err)
		}

		// negative vocab size is hacky way of signaling unshared weights. bit yikes.
		var shared_weights int32
		if config.vocab_size > 0 {
			shared_weights = 1
		}
		config.vocab_size = int32(math.Abs(float64(config.vocab_size)))
		weights, err := NewTransformerWeights(f, config, shared_weights)
		if err != nil {
			log.Fatalln(err)
		}

		if err := f.Close(); err != nil {
			log.Fatalln(err)
		}

		if steps <= 0 || steps > int(config.seq_len) {
			steps = int(config.seq_len)
		}

		// read in the tokenizer.bin file
		vocab := make([][]byte, config.vocab_size)
		tokenizer, err := os.Open(tokenizerPath)
		if err != nil {
			log.Fatalln(err)
		}
		defer tokenizer.Close()
		for i := 0; i < int(config.vocab_size); i++ {
			var size int32
			if err := binary.Read(tokenizer, binary.LittleEndian, &size); err != nil {
				log.Fatalln(err)
			}
			vocab[i] = make([]byte, size)
			_, err := tokenizer.Read(vocab[i])
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatalln(err)
			}
		}

		if err := tokenizer.Close(); err != nil {
			log.Fatalln(err)
		}

		state := NewRunState(config)

		start := time.Now().UnixMilli()
		var next, pos int
		var token int = 1
		fmt.Printf("<s>\n")
		for pos < steps {
			transformer(token, pos, config, state, weights)

			if temperature == 0.0 {
				next = argmax(state.logits, int(config.vocab_size))
			} else {
				for i := 0; i < len(state.logits); i++ {
					state.logits[i] /= temperature
				}
				softmax(state.logits, int(config.vocab_size))
				next = sample(state.logits, int(config.vocab_size))
			}
			fmt.Printf("%s", vocab[next])

			token = next
			pos++
		}
		end := time.Now().UnixMilli()
		fmt.Printf("\nachieved tok/s: %f", float64(config.seq_len)/float64(end-start)*1000.0)
	},
}

func init() {
	runCmd.Flags().StringP("model", "m", "./model.bin", "model checkpoint file")
	runCmd.Flags().StringP("tokenizer", "t", "./tokenizer.bin", "tokenizer file")
	runCmd.Flags().Float32P("temperature", "T", 0.9, "temperature for sampling")
	runCmd.Flags().IntP("steps", "s", 256, "max number of steps to run for")
	rootCmd.AddCommand(runCmd)
}

type Config struct {
	dim        int32 // transformer dimension
	hidden_dim int32 // for ffn layers
	n_layers   int32 // number of layers
	n_heads    int32 // number of query heads
	n_kv_heads int32 // number of key/value heads (can be < query heads because of multiquery)
	vocab_size int32 // vocabulary size, usually 256 (byte-level)
	seq_len    int32 // max sequence length
}

func NewConfig(f *os.File) (*Config, error) {
	var c Config
	if err := binary.Read(f, binary.LittleEndian, &c.dim); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &c.hidden_dim); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &c.n_layers); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &c.n_heads); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &c.n_kv_heads); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &c.vocab_size); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &c.seq_len); err != nil {
		return nil, err
	}
	return &c, nil
}

type TransformerWeights struct {
	// token embedding table
	token_embedding_table []float32 // (vocab_size, dim)
	// weights for rmsnorms
	rms_att_weight []float32 // (layer, dim) rmsnorm weights
	rms_ffn_weight []float32 // (layer, dim)
	// weights for matmuls
	wq []float32 // (layer, dim, dim)
	wk []float32 // (layer, dim, dim)
	wv []float32 // (layer, dim, dim)
	wo []float32 // (layer, dim, dim)
	// weights for ffn
	w1 []float32 // (layer, hidden_dim, dim)
	w2 []float32 // (layer, dim, hidden_dim)
	w3 []float32 // (layer, hidden_dim, dim)
	// final rmsnorm
	rms_final_weight []float32 // (dim,)
	// freq_cis for RoPE relatively positional embeddings
	freq_cis_real []float32 // (seq_len, dim/2)
	freq_cis_imag []float32 // (seq_len, dim/2)
	// (optional) classifier weights for the logits, on the last layer
	wcls []float32
}

func NewTransformerWeights(f *os.File, p *Config, shared_weights int32) (*TransformerWeights, error) {
	t := TransformerWeights{
		token_embedding_table: make([]float32, p.vocab_size*p.dim),
		rms_att_weight:        make([]float32, p.n_layers*p.dim),
		rms_ffn_weight:        make([]float32, p.n_layers*p.dim),
		wq:                    make([]float32, p.n_layers*p.dim*p.dim),
		wk:                    make([]float32, p.n_layers*p.dim*p.dim),
		wv:                    make([]float32, p.n_layers*p.dim*p.dim),
		wo:                    make([]float32, p.n_layers*p.dim*p.dim),
		w1:                    make([]float32, p.n_layers*p.dim*p.hidden_dim),
		w2:                    make([]float32, p.n_layers*p.hidden_dim*p.dim),
		w3:                    make([]float32, p.n_layers*p.dim*p.hidden_dim),
		rms_final_weight:      make([]float32, p.dim),
		freq_cis_real:         make([]float32, p.seq_len*(p.dim/p.n_heads)/2),
		freq_cis_imag:         make([]float32, p.seq_len*(p.dim/p.n_heads)/2),
	}
	if err := binary.Read(f, binary.LittleEndian, &t.token_embedding_table); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.rms_att_weight); err != nil {
		return nil, err
	}

	if err := binary.Read(f, binary.LittleEndian, &t.wq); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.wk); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.wv); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.wo); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.rms_ffn_weight); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.w1); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.w2); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.w3); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.rms_final_weight); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.freq_cis_real); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &t.freq_cis_imag); err != nil {
		return nil, err
	}
	if shared_weights == 1 {
		t.wcls = t.token_embedding_table
	} else {
		t.wcls = make([]float32, p.vocab_size*p.dim)
		if err := binary.Read(f, binary.LittleEndian, &t.wcls); err != nil {
			return nil, err
		}
	}
	return &t, nil
}

type RunState struct {
	// current wave of activations
	x      []float32 // activation at current time stamp (dim,)
	xb     []float32 // same, but inside a residual branch (dim,)
	xb2    []float32 // an additional buffer just for convenience (dim,)
	hb     []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	hb2    []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	q      []float32 // query (dim,)
	k      []float32 // key (dim,)
	v      []float32 // value (dim,)
	att    []float32 // buffer for scores/attention values (n_heads, seq_len)
	logits []float32 // output logits
	// kv cache
	key_cache   []float32 // (layer, seq_len, dim)
	value_cache []float32 // (layer, seq_len, dim)
}

func NewRunState(p *Config) *RunState {
	return &RunState{
		x:           make([]float32, p.dim),
		xb:          make([]float32, p.dim),
		xb2:         make([]float32, p.dim),
		hb:          make([]float32, p.hidden_dim),
		hb2:         make([]float32, p.hidden_dim),
		q:           make([]float32, p.dim),
		k:           make([]float32, p.dim),
		v:           make([]float32, p.dim),
		att:         make([]float32, p.n_heads*p.seq_len),
		logits:      make([]float32, p.vocab_size),
		key_cache:   make([]float32, p.n_layers*p.seq_len*p.dim),
		value_cache: make([]float32, p.n_layers*p.seq_len*p.dim),
	}
}

func accum(a, b []float32, size int) {
	for i := 0; i < size; i++ {
		a[i] += b[i]
	}
}

func rmsnorm(o, x, weight []float32, size int) {
	// calculate sum of squares
	var ss float32
	for i := 0; i < size; i++ {
		ss += x[i] * x[i]
	}
	ss /= float32(size)
	ss += 1e-5
	ss = 1 / float32(math.Sqrt(float64(ss)))
	// normalize and scale
	for i := 0; i < size; i++ {
		o[i] = weight[i] * (ss * x[i])
	}
}

func softmax(x []float32, size int) {
	// find max value (for numerical stability)
	max_val := x[0]
	for i := 1; i < size; i++ {
		if x[i] > max_val {
			max_val = x[i]
		}
	}
	// exp and sum
	var sum float32
	for i := 0; i < size; i++ {
		x[i] = float32(math.Exp(float64(x[i] - max_val)))
		sum += x[i]
	}
	// normalize
	for i := 0; i < size; i++ {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32, n, d int) {
	// W (d,n) @ x (n,) -> xout (d,)
	for i := 0; i < d; i++ {
		var val float32
		for j := 0; j < n; j++ {
			val += w[i*n+j] * x[j]
		}
		xout[i] = val
	}
}

func transformer(token, pos int, p *Config, s *RunState, w *TransformerWeights) {
	// a few convenience variables
	x := s.x
	dim := int(p.dim)
	hidden_dim := int(p.hidden_dim)
	head_size := int(p.dim / p.n_heads)

	// copy the token embedding into x
	content_row := w.token_embedding_table[token*dim : (token+1)*dim]
	copy(x, content_row)

	// pluck out the "pos" row of freq_cis_real and freq_cis_imag
	freq_cis_real_row := w.freq_cis_real[pos*head_size/2:]
	freq_cis_imag_row := w.freq_cis_imag[pos*head_size/2:]

	// forward all the layers
	for l := 0; l < int(p.n_layers); l++ {

		// attention rmsnorm
		rmsnorm(s.xb, x, w.rms_att_weight[l*dim:], dim)

		// qkv matmuls for this position
		matmul(s.q, s.xb, w.wq[l*dim*dim:], dim, dim)
		matmul(s.k, s.xb, w.wk[l*dim*dim:], dim, dim)
		matmul(s.v, s.xb, w.wv[l*dim*dim:], dim, dim)

		// apply RoPE rotation to the q and k vectors for each head
		for h := 0; h < int(p.n_heads); h++ {
			// get the q and k vectors for this head
			q := s.q[h*head_size:]
			k := s.k[h*head_size:]
			// rotate q and k by the freq_cis_real and freq_cis_imag
			for i := 0; i < head_size; i += 2 {
				q0 := q[i]
				q1 := q[i+1]
				k0 := k[i]
				k1 := k[i+1]
				fcr := freq_cis_real_row[i/2]
				fci := freq_cis_imag_row[i/2]
				q[i] = q0*fcr - q1*fci
				q[i+1] = q0*fci + q1*fcr
				k[i] = k0*fcr - k1*fci
				k[i+1] = k0*fci + k1*fcr
			}
		}

		// save key, value at this time step (pos) to our kv cache
		loff := l * int(p.seq_len) * dim // kv cache layer offset for convenience
		key_cache_row := s.key_cache[loff+pos*dim : loff+(pos+1)*dim]
		value_cache_row := s.value_cache[loff+pos*dim : loff+(pos+1)*dim]
		copy(key_cache_row, s.k)
		copy(value_cache_row, s.v)

		// multihead attention. iterate over all heads
		for h := 0; h < int(p.n_heads); h++ {
			// get the query vector for this head
			q := s.q[h*head_size:]
			// attention scores for this head
			att := s.att[h*int(p.seq_len):]
			// iterate over all timesteps, including the current one
			for t := 0; t <= pos; t++ {
				// get the key vector for this head and at this timestep
				k := s.key_cache[loff+t*dim+h*head_size:]
				// calculate the attention score as the dot product of q and k
				score := float32(0.0)
				for i := 0; i < head_size; i++ {
					score += q[i] * k[i]
				}
				score /= float32(math.Sqrt(float64(head_size)))
				// save the score to the attention buffer
				att[t] = score
			}

			// softmax the scores to get attention weights, from 0..pos inclusively
			softmax(att, pos+1)

			// weighted sum of the values, store back into xb
			for i := 0; i < head_size; i++ {
				val := float32(0.0)
				for t := 0; t <= pos; t++ {
					val += att[t] * s.value_cache[loff+t*dim+h*head_size+i] // note bad locality
				}
				s.xb[h*head_size+i] = val
			}
		}

		// final matmul to get the output of the attention
		matmul(s.xb2, s.xb, w.wo[l*dim*dim:], dim, dim)

		// residual connection back into x
		accum(x, s.xb2, dim)

		// ffn rmsnorm
		rmsnorm(s.xb, x, w.rms_ffn_weight[l*dim:], dim)

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		matmul(s.hb, s.xb, w.w1[l*dim*hidden_dim:], dim, hidden_dim)
		matmul(s.hb2, s.xb, w.w3[l*dim*hidden_dim:], dim, hidden_dim)

		// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
		for i := 0; i < hidden_dim; i++ {
			s.hb[i] = s.hb[i] * (1.0 / float32(1.0+math.Exp(-float64(s.hb[i]))))
		}

		// elementwise multiply with w3(x)
		for i := 0; i < hidden_dim; i++ {
			s.hb[i] = s.hb[i] * s.hb2[i]
		}

		// final matmul to get the output of the ffn
		matmul(s.xb, s.hb, w.w2[l*dim*hidden_dim:], hidden_dim, dim)

		// residual connection
		accum(x, s.xb, dim)
	}

	// final rmsnorm
	rmsnorm(x, x, w.rms_final_weight, dim)

	// classifier into logits
	matmul(s.logits, x, w.wcls, int(p.dim), int(p.vocab_size))
}

func sample(probabilities []float32, n int) int {
	// Sample index from probabilities, they must sum to 1
	r := rand.Float32()
	cdf := float32(0.0)
	for i := 0; i < n; i++ {
		cdf += probabilities[i]
		if r < cdf {
			return i
		}
	}
	return n - 1 // in case of rounding errors
}

func argmax(v []float32, n int) int {
	// return argmax of v in elements 0..n
	index := 0
	max := v[0]
	for i := 1; i < n; i++ {
		if v[i] > max {
			max = v[i]
			index = i
		}
	}
	return index
}
