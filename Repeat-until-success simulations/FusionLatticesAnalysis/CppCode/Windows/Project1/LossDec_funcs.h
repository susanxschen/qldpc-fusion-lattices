#pragma once

# define data_type bool

extern "C" {
    __declspec(dllexport) void LossDecoder_GaussElimin(data_type*, int, int, int);
    __declspec(dllexport) void LossDecoder_GaussElimin_print(data_type*, int, int, int);
    __declspec(dllexport) void SwitchRows(data_type*, int, int, int, int);
    __declspec(dllexport) void SubtractRows(data_type*, int, int, int, int);

    __declspec(dllexport) void LossDecoder_GaussElimin_trackqbts(data_type*, int*, int, int, int);
    __declspec(dllexport) void LossDecoder_GaussElimin_noordered_trackqbts(data_type*, int*, int*, int, int, int);
    __declspec(dllexport) void LossDecoder_GaussElimin_trackqbts_noorderedlost(data_type*, int*, int, int, int*, int);
    __declspec(dllexport) void SwitchRows_trackqbts(data_type*, int*, int, int, int, int);
    __declspec(dllexport) void SubtractRows_trackqbts(data_type*, int*, int, int, int, int);

    __declspec(dllexport) void PrintMatrix_toTerminal(data_type*, int, int);
    __declspec(dllexport) void PrintMatrix_int_toTerminal(int*, int, int);
}
