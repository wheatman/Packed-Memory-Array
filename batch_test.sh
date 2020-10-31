
#uncompressed
N=100000000

echo compressed parallel
make build/basic_uint64_t_delta_compressed_Eytzinger -B PARLAY=1
for M in 10000000 1000000 100000 10000 1000 100 10 1
do 
for WORKERS in 128 64 32 16 8 4 2
do
echo batch size = $M workers = $WORKERS
PARLAY_NUM_THREADS=$WORKERS ./build/basic_uint64_t_delta_compressed_Eytzinger batch $N $M
done
done





echo uncompressed parallel
make build/basic_uint64_t_uncompressed_Eytzinger -B PARLAY=1
for M in 10000000 1000000 100000 10000 1000 100 10 1
do 
for WORKERS in 128 64 32 16 8 4 2
do
echo batch size = $M workers = $WORKERS
PARLAY_NUM_THREADS=$WORKERS ./build/basic_uint64_t_uncompressed_Eytzinger batch $N $M
done
done

echo uncompressed serial
make build/basic_uint64_t_uncompressed_Eytzinger -B
for M in 10000000 1000000 100000 10000 1000 100 10 1
do 
echo batch size = $M workers = 1
./build/basic_uint64_t_uncompressed_Eytzinger batch $N $M
done


echo compressed serial
make build/basic_uint64_t_delta_compressed_Eytzinger -B
for M in 10000000 1000000 100000 10000 1000 100 10 1
do 
echo batch size = $M workers = 1
./build/basic_uint64_t_delta_compressed_Eytzinger batch $N $M
done
