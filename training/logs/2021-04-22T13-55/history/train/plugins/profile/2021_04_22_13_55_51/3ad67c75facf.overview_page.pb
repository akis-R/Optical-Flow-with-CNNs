?  *	?C???99A2?
^Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::FlatMap[0]::Generator?uŌ?љ@!<??1?X@)?uŌ?љ@1<??1?X@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??V`?ԙ@!m?t???X@)??d??1#? _"???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchkH?c?C??!3?oBX?V?)kH?c?C??13?oBX?V?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??<I?f??!T?`6?a?)??9]??1?0??'4J?:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::FlatMap??2??љ@!???X@)??@?mx?1??[ߏ?7?:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat
??ϻљ@!]?(#?X@)????t?1?'???3?:Preprocessing2F
Iterator::Model??lXSY??!}mɟűc?)????)o?1???v?(.?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.?"@
Conv2DBackpropFilterConv2DBackpropFilter+AL?>???!+AL?>???0">
Conv2DBackpropInputConv2DBackpropInputGm?#????!9W??????0"$
Conv2DConv2D?os????!???b{???0"
MulMuldL?gK??!(q?X?L??"
SumSumԚ??qܭ?!??x?*??"
AddNAddN?%?????!4??տ???"(
	ScatterNd	ScatterNd??*???!?i??????"
PackPack???f??!?P?)!N??"
SubSub?!?}????!?!?%????"
TileTile?&?!e ??!k6
??S??Y ?c?9jQ@a?OpbW>@qKg=?H@y      Y@"?	
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?49.2% of Op time on the host used eager execution. 100.0% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.