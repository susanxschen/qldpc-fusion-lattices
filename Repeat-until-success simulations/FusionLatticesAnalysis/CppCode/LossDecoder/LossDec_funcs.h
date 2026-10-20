# define data_type bool


extern "C" {
    void LossDecoder_GaussElimin(data_type*, int, int, int, int);
    void LossDecoder_GaussElimin_print(data_type*, int, int, int, int);
    void SwitchRows(data_type*, int, int, int, int);
    void SubtractRows(data_type*, int, int, int, int);

    void LossDecoder_GaussElimin_noordered(data_type*, int*, int, int, int, int);
    void LossDecoder_GaussElimin_noordered_trackqbts(data_type*, int*, int*, int, int, int, int);
    void LossDecoder_GaussElimin_trackqbts(data_type*, int*, int, int, int, int);
    void SwitchRows_trackqbts(data_type*, int*, int, int, int, int);
    void SubtractRows_trackqbts(data_type*, int*, int, int, int, int, int);

    void PrintMatrix_toTerminal(data_type*, int, int);
    void PrintMatrix_int_toTerminal(int*, int, int);
}