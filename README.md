# brainhead
A compile-time fixed-architecture header implementing neural networks in C++.

Currently supports fully connected feedforward nets only, albeit in what is hopefully a very fast way (not yet extensively benchmarked). It is very easy to add additional loss functions or activation functions, and additional connectivity structures are a reasonable hope for the near future.

Depends on Eigen (http://eigen.tuxfamily.org).

TODO: 
  Lots. Current functionality is very limited and mostly untested. Speed optimization is foremost, with additional network topologies secondary.
