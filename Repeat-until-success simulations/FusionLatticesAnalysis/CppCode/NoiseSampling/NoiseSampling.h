# define data_type bool

//////// To generate a .so library with Linux from this code run:
//////// g++ -c -fPIC foo.cpp -o foo.o
//////// g++ -shared -Wl,-soname,libfoo.so -o libfoo.so  foo.o
//////// where "foo" and "libfoo" are renamed accordingly


extern "C" {
    void SampleFusionLoss_passive(int, int, int*, bool*, bool*, float, float, float, bool);
    void SampleFusionLoss_simpleactive(int, int, int*, int*, int*, bool*, bool*, int*, bool*, float, float, float, bool);

    void simple_update_bias_and_singles_after_fail(int, bool, bool*, bool*, int*, int*);

    void SampleFusionErrors_uncorrelated(int, int*, int, int*, bool*, bool*, bool*, bool*, float, float);

    void multiedge_errorprob_uncorrelated(int, int, int*, bool, float*, bool*, bool*, float, float);

    void propagate_errors_multiedge(int, int, int*, int*, bool*, bool*, bool*); 

    void PrintMatrix_toTerminal(data_type*, int, int);
    void PrintMatrix_int_toTerminal(int*, int, int);
    bool randomBoolean(float);
}