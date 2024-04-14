OPT?=fast
VALGRIND?=0
INFO?=0
ENABLE_TRACE_TIMER?=0
CYCLE_TIMER?=0
DEBUG?=0
CILK?=0
PARLAY?=0
SANITIZE?=0
VQSORT?=1
ENABLE_TRACE_COUNTER?=0
DEBUG_SYMBOLS?=0
STORE_DENSITY?=0
SUPPORT_RANK?=0
EXTRA_WARNINGS?=0

ifeq ($(DEBUG),1)
DEBUG_SYMBOLS = 1
endif

ifeq ($(CILK),1)
PARLAY=0
endif

ifeq ($(VQSORT),0)
$(warning sorting and thus batch inserts is much faster with vqsort)
endif


CFLAGS := -Wall -Wextra -O$(OPT)  -std=c++20 -Iinclude -IParallelTools/ -Itlx/ -Wno-deprecated-declarations -Iparlaylib/include/ -IStructOfArrays/include/ -IEdgeMapVertexMap/include/ -Wshadow

ifeq ($(DEBUG_SYMBOLS),1)
CFLAGS += -g -gdwarf-4
endif

ifeq ($(EXTRA_WARNINGS),1)
CFLAGS += -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-old-style-cast -Wno-implicit-int-float-conversion -Wno-missing-prototypes -Wno-sign-conversion -Wno-float-conversion -Wno-missing-variable-declarations -Wno-exit-time-destructors -Wno-cast-align -Wno-ctad-maybe-unsupported
endif

ifeq ($(SANITIZE),1)
ifeq ($(CILK),1)
CFLAGS += -fsanitize=cilk,undefined,address -fno-omit-frame-pointer
# CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
else
CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
endif
endif

LDFLAGS := -lpthread

ifneq (,$(findstring g++,$(CXX)))
VQSORT = 0
endif


ifeq ($(VQSORT),1)
LDFLAGS += -lhwy -lhwy_contrib 
endif
# -ljemalloc

ifeq ($(VALGRIND),0)
CFLAGS += -march=native
endif

DEFINES := -DENABLE_TRACE_TIMER=$(ENABLE_TRACE_TIMER) -DCYCLE_TIMER=$(CYCLE_TIMER) -DCILK=$(CILK) -DPARLAY=$(PARLAY) -DDEBUG=$(DEBUG) -DVQSORT=$(VQSORT) -DENABLE_TRACE_COUNTER=$(ENABLE_TRACE_COUNTER) -DSTORE_DENSITY=$(STORE_DENSITY) -DSUPPORT_RANK=$(SUPPORT_RANK)

ifeq ($(CILK),1)
CFLAGS += -fopencilk -DPARLAY_CILK
ONE_WORKER = CILK_NWORKERS=1
else
ifeq ($(PARLAY),0)
CFLAGS += -DPARLAY_SEQUENTIAL
endif
endif


ifeq ($(DEBUG),0)
CFLAGS += -DNDEBUG
endif


ifeq ($(INFO), 1) 
# CFLAGS +=  -Rpass-missed="(inline|loop*)" 
#CFLAGS += -Rpass="(inline|loop*)" -Rpass-missed="(inline|loop*)" -Rpass-analysis="(inline|loop*)" 
CFLAGS += -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.* 
endif

.PHONY: all test basic 

define get_build_rule

build/basic_$(1)_$(2)_$(3): run.cpp include/PMA/internal/test.hpp include/PMA/internal/leaf.hpp include/PMA/CPMA.hpp
	@mkdir -p build
	$(CXX) $(CFLAGS) $(DEFINES) -DKEY_TYPE=$(1) -DLEAFFORM=$(2) -DHEADFORM=$(3)  -o build/basic_$(1)_$(2)_$(3) run.cpp $(LDFLAGS)
endef

define get_build_rule_name
build/basic_$(1)_$(2)_$(3)
endef

define get_build_soa_rule

build/basic_soa_$(1)_$(2): run_soa.cpp include/PMA/internal/test_map.hpp include/PMA/internal/leaf.hpp include/PMA/CPMA.hpp
	@mkdir -p build
	$(CXX) $(CFLAGS) $(DEFINES) -DKEY_TYPE=$(1) -DHEADFORM=$(2)   -o build/basic_soa_$(1)_$(2) run_soa.cpp $(LDFLAGS)
endef

define get_build_soa_rule_name
build/basic_soa_$(1)_$(2)
endef

define get_test_rule

test_out/basic_$(1)_$(2)_$(3)/v_leaf: build/basic_$(1)_$(2)_$(3)
	@mkdir -p test_out/basic_$(1)_$(2)_$(3)
	@stdbuf --output=L ./build/basic_$(1)_$(2)_$(3) v_leaf > test_out/basic_$(1)_$(2)_$(3)/v_leaf_  || (echo "verification test_leaf failed $$?"; echo "run with ./build/basic_$(1)_$(2)_$(3) v_leaf"; exit 1)
	@mv test_out/basic_$(1)_$(2)_$(3)/v_leaf_ test_out/basic_$(1)_$(2)_$(3)/v_leaf 
	@echo "test leaf passed for " build/basic_$(1)_$(2)_$(3)

test_out/basic_$(1)_$(2)_$(3)/v_batch: build/basic_$(1)_$(2)_$(3)
	@mkdir -p test_out/basic_$(1)_$(2)_$(3)
	@stdbuf --output=L ./build/basic_$(1)_$(2)_$(3) v_batch > test_out/basic_$(1)_$(2)_$(3)/v_batch_  || (echo "verification test_batch failed $$?"; echo "run with ./build/basic_$(1)_$(2)_$(3) v_batch"; exit 1)
	@stdbuf --output=L ./build/basic_$(1)_$(2)_$(3) v_batch 100000 1 >> test_out/basic_$(1)_$(2)_$(3)/v_batch_  || (echo "verification test_batch sorted failed $$?"; echo "run with ./build/basic_$(1)_$(2)_$(3) v_batch  100000 1"; exit 1)
	@mv test_out/basic_$(1)_$(2)_$(3)/v_batch_ test_out/basic_$(1)_$(2)_$(3)/v_batch
	@echo "test batch passed for " build/basic_$(1)_$(2)_$(3)

test_out/basic_$(1)_$(2)_$(3)/verify: build/basic_$(1)_$(2)_$(3)
	@mkdir -p test_out/basic_$(1)_$(2)_$(3)
	@stdbuf --output=L ./build/basic_$(1)_$(2)_$(3) verify > test_out/basic_$(1)_$(2)_$(3)/verify_  || (echo "verification test verify failed $$?"; echo "run with ./build/basic_$(1)_$(2)_$(3) verify"; exit 1)
	@mv test_out/basic_$(1)_$(2)_$(3)/verify_ test_out/basic_$(1)_$(2)_$(3)/verify
	@echo "test verify passed for " build/basic_$(1)_$(2)_$(3)

test_out/basic_$(1)_$(2)_$(3)/test: test_out/basic_$(1)_$(2)_$(3)/v_leaf test_out/basic_$(1)_$(2)_$(3)/v_batch test_out/basic_$(1)_$(2)_$(3)/verify
	@touch test_out/basic_$(1)_$(2)_$(3)/test
endef

define get_test_rule_name
test_out/basic_$(1)_$(2)_$(3)/test
endef

define get_test_soa_rule

test_out/basic_soa_$(1)_$(2)/verify: build/basic_soa_$(1)_$(2)
	@mkdir -p test_out/basic_soa_$(1)_$(2)
	@stdbuf --output=L ./build/basic_soa_$(1)_$(2) verify > test_out/basic_soa_$(1)_$(2)/verify_  || (echo "verification test verify failed $$?"; echo "run with ./build/basic_soa_$(1)_$(2) verify"; exit 1)
	@mv test_out/basic_soa_$(1)_$(2)/verify_ test_out/basic_soa_$(1)_$(2)/verify
	@echo "test verify passed for " build/basic_soa_$(1)_$(2)

test_out/basic_soa_$(1)_$(2)/test: test_out/basic_soa_$(1)_$(2)/verify
	@touch test_out/basic_soa_$(1)_$(2)/test

endef

define get_test_soa_rule_name
test_out/basic_soa_$(1)_$(2)/test
endef

define get_build_rule_pcsr

build/pcsr_$(1)_$(2)_$(3): run_pcsr.cpp include/PMA/internal/leaf.hpp include/PMA/CPMA.hpp include/PMA/PCSR.hpp
	@mkdir -p build
	$(CXX) $(CFLAGS) $(DEFINES) -DKEY_TYPE=$(1) -DLEAFFORM=$(2) -DHEADFORM=$(3) -o build/pcsr_$(1)_$(2)_$(3) run_pcsr.cpp $(LDFLAGS)
endef

define get_build_rule_name_pcsr
build/pcsr_$(1)_$(2)_$(3)
endef

define get_test_pcsr_rule

test_out/pcsr_$(1)_$(2)_$(3)/graph: build/pcsr_$(1)_$(2)_$(3)
	@mkdir -p test_out/prcs_$(1)_$(2)_$(3)
	./build/pcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot.adj 1 1 > /dev/null
	@diff -q bfs.out EdgeMapVertexMap/correct_output/slashdot/bfs/source1 
	@diff -q bc.out EdgeMapVertexMap/correct_output/slashdot/bc/source1 
	@diff -q pr.out EdgeMapVertexMap/correct_output/slashdot/pr/iters10 
	@diff -q cc.out EdgeMapVertexMap/correct_output/slashdot/cc/output 
	./build/pcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot.adj 1 2 > /dev/null
	@diff -q bfs.out EdgeMapVertexMap/correct_output/slashdot/bfs/source2
	@diff -q bc.out EdgeMapVertexMap/correct_output/slashdot/bc/source2
	./build/pcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot.adj 1 3 > /dev/null
	@diff -q bfs.out EdgeMapVertexMap/correct_output/slashdot/bfs/source3
	@diff -q bc.out EdgeMapVertexMap/correct_output/slashdot/bc/source3
	./build/pcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot.adj 1 4 > /dev/null
	@diff -q bfs.out EdgeMapVertexMap/correct_output/slashdot/bfs/source4
	@diff -q bc.out EdgeMapVertexMap/correct_output/slashdot/bc/source4
	./build/pcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot.adj 1 5 > /dev/null
	@diff -q bfs.out EdgeMapVertexMap/correct_output/slashdot/bfs/source5
	@diff -q bc.out EdgeMapVertexMap/correct_output/slashdot/bc/source5
endef

define get_test_pcsr_rule_name
test_out/pcsr_$(1)_$(2)_$(3)/graph
endef

define get_build_rule_wpcsr

build/wpcsr_$(1)_$(2)_$(3): run_wpcsr.cpp include/PMA/internal/leaf.hpp include/PMA/CPMA.hpp include/PMA/PCSR.hpp
	@mkdir -p build
	$(CXX) $(CFLAGS) $(DEFINES) -DKEY_TYPE=$(1) -DWEIGHT_TYPE=$(2) -DHEADFORM=$(3)   -o build/wpcsr_$(1)_$(2)_$(3) run_wpcsr.cpp $(LDFLAGS)
endef

define get_build_rule_name_wpcsr
build/wpcsr_$(1)_$(2)_$(3)
endef

define get_test_wpcsr_rule

test_out/wpcsr_$(1)_$(2)_$(3)/graph: build/wpcsr_$(1)_$(2)_$(3)
	@mkdir -p test_out/wprcs_$(1)_$(2)_$(3)
	./build/wpcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot_weights.adj 1 1 > /dev/null
	@diff -q bf.out EdgeMapVertexMap/correct_output/slashdot_weights/bf/source1 
	./build/wpcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot_weights.adj 1 2 > /dev/null
	@diff -q bf.out EdgeMapVertexMap/correct_output/slashdot_weights/bf/source2
	./build/wpcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot_weights.adj 1 3 > /dev/null
	@diff -q bf.out EdgeMapVertexMap/correct_output/slashdot_weights/bf/source3
	./build/wpcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot_weights.adj 1 4 > /dev/null
	@diff -q bf.out EdgeMapVertexMap/correct_output/slashdot_weights/bf/source4
	./build/wpcsr_$(1)_$(2)_$(3) EdgeMapVertexMap/data/slashdot_weights.adj 1 5 > /dev/null
	@diff -q bf.out EdgeMapVertexMap/correct_output/slashdot_weights/bf/source5
endef

define get_test_wpcsr_rule_name
test_out/wpcsr_$(1)_$(2)_$(3)/graph
endef

all: basic

head_forms := InPlace Linear Eytzinger BNary
key_types := uint32_t uint64_t
leaf_forms := delta_compressed uncompressed

$(foreach leaf_form,$(leaf_forms),$(foreach key_type,$(key_types),$(foreach head_form,$(head_forms),$(eval $(call get_build_rule,$(key_type),$(leaf_form),$(head_form))))))

$(foreach leaf_form,$(leaf_forms),$(foreach key_type,$(key_types),$(foreach head_form,$(head_forms),$(eval $(call get_test_rule,$(key_type),$(leaf_form),$(head_form))))))

# 16 and 24 bit ones don't bother with compression

key_types2 := uint16_t uint24_t
$(foreach key_type,$(key_types2),$(foreach head_form,$(head_forms),$(eval $(call get_build_rule,$(key_type),uncompressed,$(head_form)))))

$(foreach key_type,$(key_types2),$(foreach head_form,$(head_forms),$(eval $(call get_test_rule,$(key_type),uncompressed,$(head_form)))))


# key values stores do all the key sizes, but don't do compression
key_types3 := uint16_t uint24_t uint32_t uint64_t

$(foreach key_type,$(key_types3),$(foreach head_form,$(head_forms),$(eval $(call get_build_soa_rule,$(key_type),$(head_form)))))

$(foreach key_type,$(key_types3),$(foreach head_form,$(head_forms),$(eval $(call get_test_soa_rule,$(key_type),$(head_form)))))


#pcsr rules
head_forms_pcsr := InPlace Linear Eytzinger
key_types_pcsr := uint32_t uint64_t
leaf_forms_pcsr :=  uncompressed

$(foreach leaf_form,$(leaf_forms_pcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(eval $(call get_build_rule_pcsr,$(key_type),$(leaf_form),$(head_form))))))

$(foreach leaf_form,$(leaf_forms_pcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(eval $(call get_test_pcsr_rule,$(key_type),$(leaf_form),$(head_form))))))

weight_types_wpcsr := uint16_t uint32_t uint64_t

$(foreach weight_type,$(weight_types_wpcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(eval $(call get_build_rule_wpcsr,$(key_type),$(weight_type),$(head_form))))))

$(foreach weight_type,$(weight_types_wpcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(eval $(call get_test_wpcsr_rule,$(key_type),$(weight_type),$(head_form))))))



basic : $(foreach leaf_form,$(leaf_forms),$(foreach key_type,$(key_types),$(foreach head_form,$(head_forms),$(call get_build_rule_name,$(key_type),$(leaf_form),$(head_form))))) $(foreach key_type,$(key_types2),$(foreach head_form,$(head_forms),$(call get_build_rule_name,$(key_type),uncompressed,$(head_form)))) $(foreach key_type,$(key_types3),$(foreach head_form,$(head_forms),$(call get_build_soa_rule_name,$(key_type),$(head_form)))) 

test : $(foreach leaf_form,$(leaf_forms),$(foreach key_type,$(key_types),$(foreach head_form,$(head_forms),$(call get_test_rule_name,$(key_type),$(leaf_form),$(head_form))))) $(foreach key_type,$(key_types2),$(foreach head_form,$(head_forms),$(call get_test_rule_name,$(key_type),uncompressed,$(head_form)))) $(foreach key_type,$(key_types3),$(foreach head_form,$(head_forms),$(call get_test_soa_rule_name,$(key_type),$(head_form))))


soa : run_soa.cpp include/PMA/internal/leaf.hpp include/PMA/CPMA.hpp StructOfArrays/soa.hpp
	$(CXX) $(CFLAGS) $(DEFINES) -o soa run_soa.cpp $(LDFLAGS)
 



pcsr : $(foreach leaf_form,$(leaf_forms_pcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(call get_build_rule_name_pcsr,$(key_type),$(leaf_form),$(head_form)))))
pcsr_test : $(foreach leaf_form,$(leaf_forms_pcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(call get_test_pcsr_rule_name,$(key_type),$(leaf_form),$(head_form)))))

wpcsr : $(foreach weight_type,$(weight_types_wpcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(call get_build_rule_name_wpcsr,$(key_type),$(weight_type),$(head_form)))))
wpcsr_test : $(foreach weight_type,$(weight_types_wpcsr),$(foreach key_type,$(key_types_pcsr),$(foreach head_form,$(head_forms_pcsr),$(call get_test_wpcsr_rule_name,$(key_type),$(weight_type),$(head_form)))))


clean:
	rm -f *.o code.profdata default.profraw
	rm -rf build/* test_out/*
