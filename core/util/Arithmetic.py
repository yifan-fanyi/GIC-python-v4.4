# ppm mode not correct yet
__version__ = "2.5.0"
from core.util import myLog
myLog('<Framework Version> core.Arithmetic -> %s'%__version__)
print('<Framework Version> core.Arithmetic -> %s'%__version__)
import os
import numpy as np
import contextlib
import core.util.arithmeticcoding as arithmeticcoding

class Arithmetic:
	def __init__(self, mode='fix', n_symbols=-1, num_state_bits=32):
		self.mode = mode
		self.freqs = None
		self.n_symbols = n_symbols
		self.num_state_bits = num_state_bits
		self.MODEL_ORDER = 3

	def fit(self, x):
		x = np.array(x).reshape(-1).astype('int16')
		self.freqs = arithmeticcoding.SimpleFrequencyTable([0] * (self.n_symbols+1))
		for i in range(x.shape[0]):
			self.freqs.increment(x[i])
		self.freqs.increment(self.n_symbols)
		return self

	def encode_fix(self, x, outfile='./cache/tmp.ac'):
		with contextlib.closing(arithmeticcoding.BitOutputStream(open(outfile, "wb"))) as bitout:
			enc = arithmeticcoding.ArithmeticEncoder(self.num_state_bits, bitout)
			for i in range(x.shape[0]):
				enc.write(self.freqs, x[i])
			enc.write(self.freqs, self.n_symbols) 
			enc.finish()
		#print('   Arithmetic: average bits/symbol: %3.4f'%(os.stat(outfile).st_size*8/float(len(x))))
		s = ''
		for i in range(os.stat(outfile).st_size*8):
			s+='0'
		return s

	def decode_fix(self, outfile='./cache/tmp.ac'):
		res= []
		with open(outfile, "rb") as inp:
			bitin = arithmeticcoding.BitInputStream(inp)
			dec = arithmeticcoding.ArithmeticDecoder(self.num_state_bits, bitin)
			while True:
				symbol = dec.read(self.freqs)
				if symbol == self.n_symbols:  # EOF symbol
					break
				res.append(symbol)
		return np.array(res)

	def encode_adp(self, x, outfile='./cache/tmp.ac', init=True):
		with contextlib.closing(arithmeticcoding.BitOutputStream(open(outfile, "wb"))) as bitout:
			if init == True:
				initfreqs = arithmeticcoding.FlatFrequencyTable(self.n_symbols+1)
				self.freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
			enc = arithmeticcoding.ArithmeticEncoder(self.num_state_bits, bitout)
			for i in range(len(x)):
				enc.write(self.freqs, x[i])
				self.freqs.increment(x[i])
			enc.write(self.freqs, self.n_symbols)
			enc.finish()  # Flush remaining code bits
		print('   Arithmetic adp: average bits/symbol: %3.4f'%(os.stat(outfile).st_size*8/float(len(x))))
		s = ''
		for i in range(os.stat(outfile).st_size*8):
			s+='0'
		return s
                 

	def decode_adp(self, outfile='./cache/tmp.ac', init=True):
		res = []
		with open(outfile, "rb") as inp:
			bitin = arithmeticcoding.BitInputStream(inp)
			if init == True:
				initfreqs = arithmeticcoding.FlatFrequencyTable(self.n_symbols+1)
				self.freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
			dec = arithmeticcoding.ArithmeticDecoder(self.num_state_bits, bitin)
			while True:
				symbol = dec.read(self.freqs)
				if symbol == self.n_symbols:  # EOF symbol
					break
				res.append(symbol)
				self.freqs.increment(symbol)
		return np.array(res)

	def encode_ppm(self, x, outfile):
		def encode_symbol(model, history, symbol, enc):
			for order in reversed(range(len(history) + 1)):
				ctx = model.root_context
				for sym in history[ : order]:
					assert ctx.subcontexts is not None
					ctx = ctx.subcontexts[sym]
					if ctx is None:
						break
				else:  # ctx is not None
					if symbol != self.n_symbols and ctx.frequencies.get(symbol) > 0:
						enc.write(ctx.frequencies, symbol)
						return
					# Else write context escape symbol and continue decrementing the order
					enc.write(ctx.frequencies, self.n_symbols)
			# Logic for order = -1
			enc.write(model.order_minus1_freqs, symbol)

		with contextlib.closing(arithmeticcoding.BitOutputStream(open(outfile, "wb"))) as bitout:
			enc = arithmeticcoding.ArithmeticEncoder(self.num_state_bits, bitout)
			model = ppmmodel.PpmModel(self.MODEL_ORDER, self.n_symbols+1, self.n_symbols)
			history = []
			for i in range(len(x)):
				encode_symbol(model, history, x[i], enc)
				model.increment_contexts(history, x[i])
				if model.model_order >= 1:
					if len(history) == model.model_order:
						history.pop()
					history.insert(0, x[i])
			encode_symbol(model, history, self.n_symbols, enc)  # EOF
			enc.finish()  # Flush remaining code bits
		print('   Arithmetic ppm: average bits/symbol: %3.4f'%(os.stat(outfile).st_size*8/float(len(x))))
		return os.stat(outfile).st_size*8

	def decode_ppm(self, outfile):
		def decode_symbol(dec, model, history):
			for order in reversed(range(len(history) + 1)):
				ctx = model.root_context
				for sym in history[ : order]:
					assert ctx.subcontexts is not None
					ctx = ctx.subcontexts[sym]
					if ctx is None:
						break
				else:  # ctx is not None
					symbol = dec.read(ctx.frequencies)
					if symbol < 256:
						return symbol
			return dec.read(model.order_minus1_freqs)
		res = []
		with open(outfile, "rb") as inp:
			bitin = arithmeticcoding.BitInputStream(inp)
			dec = arithmeticcoding.ArithmeticDecoder(self.num_state_bits, bitin)
			model = ppmmodel.PpmModel(self.MODEL_ORDER, self.n_symbols+1, self.n_symbols)
			history = []
			while True:
				symbol = decode_symbol(dec, model, history)
				if symbol == 256:  # EOF symbol
					break
				res.append(symbol)
				model.increment_contexts(history, symbol)
				if model.model_order >= 1:
					if len(history) == model.model_order:
						history.pop()
					history.insert(0, symbol)
		return np.array(res)

	def encode(self, x, outfile='./cache/tmp.ac', init=True):
		x = np.array(x).reshape(-1).astype('int16')
		if self.mode == 'fix':
			return self.encode_fix(x, outfile)
		elif self.mode == 'adp':
			return self.encode_adp(x, outfile, init)
		elif self.mode == 'ppm':
			return self.encode_ppm(x, outfile)
		else:
			assert False, "Error encoding mode, not in (\'fix\' or \'adp\' or \'ppm\')"

	def decode(self, outfile='./cache/tmp.ac', init=True):
		if self.mode == 'fix':
			return self.decode_fix(outfile)
		elif self.mode == 'adp':
			return self.decode_adp(outfile, init)
		elif self.mode == 'ppm':
			return self.decode_ppm(outfile, init)
		else:
			assert False, "Error decoding mode, not in (\'fix\' or \'adp\' or \'ppm\')"

if __name__ == "__main__":
	a = []
	for ct in range(12, -1, -1):
		for i in range(pow(2,ct)//5):
			a.append(12-ct)
	a = np.array(a)

	ac = Arithmetic(mode='fix', n_symbols=len(np.unique(a))).fit(a)
	ac.encode(a, 'tmp.ac')
	ia = ac.decode('tmp.ac')
	print(np.sum(np.abs(a-ia)))

	ac = Arithmetic(mode='adp', n_symbols=len(np.unique(a)))
	ac.encode(a, 'tmp.ac')
	ia = ac.decode('tmp.ac')
	print(np.sum(np.abs(a-ia)))

	#ac = Arithmetic(mode='ppm', n_symbols=len(np.unique(a)))
	#ac.encode(a, 'tmp.ac')
	#ia = ac.decode('tmp.ac')
	#print(np.sum(np.abs(a-ia)))