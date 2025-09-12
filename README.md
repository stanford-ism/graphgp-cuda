# graphgp-cuda


Implementation TODOs:
 - 64 bit option, requires templating functions and copying XLA bindings, check everything is size_t
 - inverse and log determinant, with derivatives
 - make sure graph, especially tree, is correct at large N. differences probably due to tie-breaking. but something funny with block size 2048 on the 3 key sort

Code improvement:
 - clean up all the copy row indices and permute functions in `common.h`
 - concise system for automatically setting up custom derivatives for jax
 - automatic sort as a function of arbitrary number of keys
 - only do what's necessary in tree building, right now we copy too much