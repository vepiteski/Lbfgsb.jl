macro checked_lib(libname, path)
    (Libdl.dlopen_e(path) == C_NULL) && error("Unable to load \n\n$libname ($path)

Please re-run Pkg.build(package), and restart Julia.")
    quote const $(esc(libname)) = $path end
end
@checked_lib liblbfgsbf "/home/local/USHERBROOKE/dusj1701/.julia/v0.5/Lbfgsb/deps/usr/lib/liblbfgsbf.so"
