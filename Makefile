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
EXTRA_WARNINGS?=0

ifeq ($(CILK),1)
PARLAY=0
endif

ifeq ($(VQSORT),0)
$(warning sorting and thus batch inserts is much faster with vqsort)
endif


CFLAGS := -Wall -Wextra -O$(OPT)  -std=c++20 -IParallelTools/ -Itlx/ -Wno-deprecated-declarations -Iparlaylib/include/ -ferror-limit=1 -Wshadow

ifeq ($(DEBUG_SYMBOLS),1)
CFLAGS += -g -gdwarf-4
endif

ifeq ($(EXTRA_WARNINGS),1)
CFLAGS += -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-old-style-cast -Wno-implicit-int-float-conversion -Wno-missing-prototypes -Wno-sign-conversion -Wno-float-conversion -Wno-missing-variable-declarations -Wno-exit-time-destructors
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

DEFINES := -DENABLE_TRACE_TIMER=$(ENABLE_TRACE_TIMER) -DCYCLE_TIMER=$(CYCLE_TIMER) -DCILK=$(CILK) -DPARLAY=$(PARLAY) -DDEBUG=$(DEBUG) -DVQSORT=$(VQSORT) -DENABLE_TRACE_COUNTER=$(ENABLE_TRACE_COUNTER) -DSTORE_DENSITY=$(STORE_DENSITY)

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

build/basic_$(1)_$(2)_$(3): run.cpp test.hpp leaf.hpp CPMA.hpp
	@mkdir -p build
	$(CXX) $(CFLAGS) $(DEFINES) -DKEY_TYPE=$(1) -DLEAFFORM=$(2) -DHEADFORM=$(3)  $(LDFLAGS) -o build/basic_$(1)_$(2)_$(3) run.cpp
endef

define get_build_rule_name
build/basic_$(1)_$(2)_$(3)
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




basic : $(foreach leaf_form,$(leaf_forms),$(foreach key_type,$(key_types),$(foreach head_form,$(head_forms),$(call get_build_rule_name,$(key_type),$(leaf_form),$(head_form))))) $(foreach key_type,$(key_types2),$(foreach head_form,$(head_forms),$(call get_build_rule_name,$(key_type),uncompressed,$(head_form))))

test : $(foreach leaf_form,$(leaf_forms),$(foreach key_type,$(key_types),$(foreach head_form,$(head_forms),$(call get_test_rule_name,$(key_type),$(leaf_form),$(head_form))))) $(foreach key_type,$(key_types2),$(foreach head_form,$(head_forms),$(call get_test_rule_name,$(key_type),uncompressed,$(head_form))))


soa : run_soa.cpp leaf.hpp CPMA.hpp StructOfArrays/soa.hpp
	$(CXX) $(CFLAGS) $(DEFINES)  $(LDFLAGS) -o soa run_soa.cpp
 


clean:
	rm -f *.o code.profdata default.profraw
	rm -rf build/* test_out/*
