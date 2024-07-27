/FlexFlow/build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:cpu 4 -ll:csize 120000 -ll:fsize 20000 -ll:zsize 80000 -lg:eager_alloc_percentage 30 --fusion --use-full-precision -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-160m -prompt /FlexFlow/inference/prompt/test.json -output-file /FlexFlow/inference/output/spec_inference_llama.txt -tensor-parallelism-degree 4 --max-sequence-length 512 --max-requests-per-batch 5