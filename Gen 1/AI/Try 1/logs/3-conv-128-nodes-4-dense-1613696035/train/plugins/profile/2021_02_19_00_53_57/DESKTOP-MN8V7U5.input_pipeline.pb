	'1???@'1???@!'1???@	?? !&z??? !&z?!?? !&z?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$'1???@? ?	???AI??&??@Y,e?X??*	????̌L@2U
Iterator::Model::ParallelMapV2??H?}??!r???
89@)??H?}??1r???
89@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?q??????!?G ?R;@)?{??Pk??1?????6@:Preprocessing2F
Iterator::Modelp_?Q??!?[i??F@)M??St$??1_?>3?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/?$???!???؇c2@)S?!?uq{?1?W???w'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2U0*???!???~K@)????Mbp?1*??Ia@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!N?F??@)ŏ1w-!o?1N?F??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!????@)??_vOf?1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??0?*??!(??&^?4@)??_?LU?1(?p?6@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?? !&z?I}{g??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? ?	???? ?	???!? ?	???      ??!       "      ??!       *      ??!       2	I??&??@I??&??@!I??&??@:      ??!       B      ??!       J	,e?X??,e?X??!,e?X??R      ??!       Z	,e?X??,e?X??!,e?X??b      ??!       JCPU_ONLYY?? !&z?b q}{g??X@