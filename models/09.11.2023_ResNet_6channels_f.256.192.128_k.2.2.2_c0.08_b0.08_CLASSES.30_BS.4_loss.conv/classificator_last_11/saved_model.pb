��)
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��"
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0
�
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameconv2d_9/kernel
|
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*'
_output_shapes
:�*
dtype0
�
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_11/moving_variance
�
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_11/moving_mean
�
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_11/beta
�
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_11/gamma
�
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:�*
dtype0
�
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_8/moving_variance
�
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_8/moving_mean
�
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_8/beta
�
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_8/gamma
�
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:�*
dtype0
�
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_10/moving_variance
�
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_10/moving_mean
�
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_10/beta
�
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_10/gamma
�
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes	
:�*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:�*
dtype0
�
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:�*
dtype0
�
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_8/kernel
}
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*(
_output_shapes
:��*
dtype0
�
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_9/moving_variance
�
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_9/moving_mean
�
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_9/beta
�
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_9/gamma
�
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes	
:�*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:�*
dtype0
�
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_7/kernel
}
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*(
_output_shapes
:��*
dtype0
�
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_7/moving_variance
�
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_7/moving_mean
�
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_7/beta
�
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_7/gamma
�
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:�*
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_6/moving_variance
�
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_6/moving_mean
�
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_6/beta
�
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_6/gamma
�
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:�*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:�*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:�*
dtype0
�
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:��*
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:�*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:�*
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:��*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:�*
dtype0

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:�*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:�*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:��*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:�*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:�*
dtype0
�
serving_default_input_2Placeholder*A
_output_shapes/
-:+���������������������������*
dtype0*6
shape-:+���������������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d/kernelconv2d/biasconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_5/kernelconv2d_5/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_8/kernelconv2d_8/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancebatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancebatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_9/kernelconv2d_9/bias*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_338099

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�	
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer-24
layer_with_weights-15
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer_with_weights-18
layer-30
 layer_with_weights-19
 layer-31
!layer-32
"layer_with_weights-20
"layer-33
#layer_with_weights-21
#layer-34
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_default_save_signature
+
signatures*
* 
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance*
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op*
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance*
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance*
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
20
31
B2
C3
D4
E5
L6
M7
U8
V9
k10
l11
m12
n13
v14
w15
x16
y17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67*
�
20
31
B2
C3
L4
M5
U6
V7
k8
l9
v10
w11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*_default_save_signature
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 

20
31*

20
31*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
B0
C1
D2
E3*

B0
C1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

U0
V1*

U0
V1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
k0
l1
m2
n3*

k0
l1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
v0
w1
x2
y3*

v0
w1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_4/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_4/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_4/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_4/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_9/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_9/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_9/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_9/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_8/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_8/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_10/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_10/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_10/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE&batch_normalization_10/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_11/gamma6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_11/beta5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_11/moving_mean<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE&batch_normalization_11/moving_variance@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_9/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_9/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
D0
E1
m2
n3
x4
y5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

m0
n1*
* 
* 
* 
* 
* 
* 
* 
* 

x0
y1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOpConst*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_340159
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasconv2d/kernelconv2d/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_5/kernelconv2d_5/biasconv2d_3/kernelconv2d_3/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_8/kernelconv2d_8/biasconv2d_6/kernelconv2d_6/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancebatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancebatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_9/kernelconv2d_9/bias*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_340373��
�
k
A__inference_add_2_layer_call_and_return_conditional_losses_336668

inputs
inputs_1
identityk
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,����������������������������j
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336227

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335874

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
R
&__inference_add_1_layer_call_fn_339497
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_336555{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_0:lh
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_1
�
P
$__inference_add_layer_call_fn_339150
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_336442{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_0:lh
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_1
�
�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_339237

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335682

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_339694

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
'__inference_conv2d_layer_call_fn_338990

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_336382�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339218

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335746

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
k
A__inference_add_1_layer_call_and_return_conditional_losses_336555

inputs
inputs_1
identityk
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,����������������������������j
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339776

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_338890

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
1__inference_ResNet_6channels_layer_call_fn_336836
input_2"
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�$
	unknown_5:�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:	�&

unknown_43:��

unknown_44:	�

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�&

unknown_49:��

unknown_50:	�&

unknown_51:��

unknown_52:	�

unknown_53:	�

unknown_54:	�

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�%

unknown_65:�

unknown_66:
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_336697�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2
�
e
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_336474

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_339584

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_335971

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339064

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_339020

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_9_layer_call_fn_339620

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336130�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_2_layer_call_fn_339005

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_336416{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335810

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_6_layer_call_fn_339380

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335907�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339894

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
1__inference_ResNet_6channels_layer_call_fn_337608
input_2"
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�$
	unknown_5:�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:	�&

unknown_43:��

unknown_44:	�

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�&

unknown_49:��

unknown_50:	�&

unknown_51:��

unknown_52:	�

unknown_53:	�

unknown_54:	�

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�%

unknown_65:�

unknown_66:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*N
_read_only_resource_inputs0
.,	
 !"%&)*-./0345678;<?@CD*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337328�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_338981

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339638

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_339357

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_3_layer_call_fn_339362

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_336522{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336066

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_336608

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335618

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_2_layer_call_fn_339033

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335651�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_338099
input_2"
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�$
	unknown_5:�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:	�&

unknown_43:��

unknown_44:	�

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�&

unknown_49:��

unknown_50:	�&

unknown_51:��

unknown_52:	�

unknown_53:	�

unknown_54:	�

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�%

unknown_65:�

unknown_66:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_335565�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2
�	
�
6__inference_batch_normalization_9_layer_call_fn_339607

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336099�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_336587

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_336642

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
c
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_336409

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_6_layer_call_fn_339709

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_336635{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339200

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_2_layer_call_fn_338971

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_336398�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
B__inference_conv2d_layer_call_and_return_conditional_losses_339000

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_layer_call_fn_339108

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335746�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_7_layer_call_fn_339516

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336035�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339491

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336291

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339126

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_336635

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_336624

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_339010

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_339347

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336194

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_2_layer_call_fn_339046

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335682�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_8_layer_call_fn_339665

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_336624�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_5_layer_call_fn_339260

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335843�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
R
&__inference_add_2_layer_call_fn_339844
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_336668{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_0:lh
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_1
�	
�
6__inference_batch_normalization_6_layer_call_fn_339393

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335938�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�.
"__inference__traced_restore_340373
file_prefix;
 assignvariableop_conv2d_1_kernel:�/
 assignvariableop_1_conv2d_1_bias:	�=
.assignvariableop_2_batch_normalization_1_gamma:	�<
-assignvariableop_3_batch_normalization_1_beta:	�C
4assignvariableop_4_batch_normalization_1_moving_mean:	�G
8assignvariableop_5_batch_normalization_1_moving_variance:	�>
"assignvariableop_6_conv2d_2_kernel:��/
 assignvariableop_7_conv2d_2_bias:	�;
 assignvariableop_8_conv2d_kernel:�-
assignvariableop_9_conv2d_bias:	�>
/assignvariableop_10_batch_normalization_2_gamma:	�=
.assignvariableop_11_batch_normalization_2_beta:	�D
5assignvariableop_12_batch_normalization_2_moving_mean:	�H
9assignvariableop_13_batch_normalization_2_moving_variance:	�<
-assignvariableop_14_batch_normalization_gamma:	�;
,assignvariableop_15_batch_normalization_beta:	�B
3assignvariableop_16_batch_normalization_moving_mean:	�F
7assignvariableop_17_batch_normalization_moving_variance:	�>
/assignvariableop_18_batch_normalization_3_gamma:	�=
.assignvariableop_19_batch_normalization_3_beta:	�D
5assignvariableop_20_batch_normalization_3_moving_mean:	�H
9assignvariableop_21_batch_normalization_3_moving_variance:	�?
#assignvariableop_22_conv2d_4_kernel:��0
!assignvariableop_23_conv2d_4_bias:	�>
/assignvariableop_24_batch_normalization_5_gamma:	�=
.assignvariableop_25_batch_normalization_5_beta:	�D
5assignvariableop_26_batch_normalization_5_moving_mean:	�H
9assignvariableop_27_batch_normalization_5_moving_variance:	�?
#assignvariableop_28_conv2d_5_kernel:��0
!assignvariableop_29_conv2d_5_bias:	�?
#assignvariableop_30_conv2d_3_kernel:��0
!assignvariableop_31_conv2d_3_bias:	�>
/assignvariableop_32_batch_normalization_6_gamma:	�=
.assignvariableop_33_batch_normalization_6_beta:	�D
5assignvariableop_34_batch_normalization_6_moving_mean:	�H
9assignvariableop_35_batch_normalization_6_moving_variance:	�>
/assignvariableop_36_batch_normalization_4_gamma:	�=
.assignvariableop_37_batch_normalization_4_beta:	�D
5assignvariableop_38_batch_normalization_4_moving_mean:	�H
9assignvariableop_39_batch_normalization_4_moving_variance:	�>
/assignvariableop_40_batch_normalization_7_gamma:	�=
.assignvariableop_41_batch_normalization_7_beta:	�D
5assignvariableop_42_batch_normalization_7_moving_mean:	�H
9assignvariableop_43_batch_normalization_7_moving_variance:	�?
#assignvariableop_44_conv2d_7_kernel:��0
!assignvariableop_45_conv2d_7_bias:	�>
/assignvariableop_46_batch_normalization_9_gamma:	�=
.assignvariableop_47_batch_normalization_9_beta:	�D
5assignvariableop_48_batch_normalization_9_moving_mean:	�H
9assignvariableop_49_batch_normalization_9_moving_variance:	�?
#assignvariableop_50_conv2d_8_kernel:��0
!assignvariableop_51_conv2d_8_bias:	�?
#assignvariableop_52_conv2d_6_kernel:��0
!assignvariableop_53_conv2d_6_bias:	�?
0assignvariableop_54_batch_normalization_10_gamma:	�>
/assignvariableop_55_batch_normalization_10_beta:	�E
6assignvariableop_56_batch_normalization_10_moving_mean:	�I
:assignvariableop_57_batch_normalization_10_moving_variance:	�>
/assignvariableop_58_batch_normalization_8_gamma:	�=
.assignvariableop_59_batch_normalization_8_beta:	�D
5assignvariableop_60_batch_normalization_8_moving_mean:	�H
9assignvariableop_61_batch_normalization_8_moving_variance:	�?
0assignvariableop_62_batch_normalization_11_gamma:	�>
/assignvariableop_63_batch_normalization_11_beta:	�E
6assignvariableop_64_batch_normalization_11_moving_mean:	�I
:assignvariableop_65_batch_normalization_11_moving_variance:	�>
#assignvariableop_66_conv2d_9_kernel:�/
!assignvariableop_67_conv2d_9_bias:
identity_69��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9� 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*S
dtypesI
G2E[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_1_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_1_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv2d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_2_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_2_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_batch_normalization_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_batch_normalization_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp3assignvariableop_16_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_3_moving_meanIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_3_moving_varianceIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_4_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_4_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_5_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_5_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_5_moving_meanIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_5_moving_varianceIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_5_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv2d_5_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_3_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_3_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_6_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_6_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_6_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_6_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_4_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_4_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_4_moving_meanIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_4_moving_varianceIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp/assignvariableop_40_batch_normalization_7_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp.assignvariableop_41_batch_normalization_7_betaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_batch_normalization_7_moving_meanIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp9assignvariableop_43_batch_normalization_7_moving_varianceIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp#assignvariableop_44_conv2d_7_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp!assignvariableop_45_conv2d_7_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp/assignvariableop_46_batch_normalization_9_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp.assignvariableop_47_batch_normalization_9_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp5assignvariableop_48_batch_normalization_9_moving_meanIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp9assignvariableop_49_batch_normalization_9_moving_varianceIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp#assignvariableop_50_conv2d_8_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp!assignvariableop_51_conv2d_8_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp#assignvariableop_52_conv2d_6_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp!assignvariableop_53_conv2d_6_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_10_gammaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp/assignvariableop_55_batch_normalization_10_betaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp6assignvariableop_56_batch_normalization_10_moving_meanIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp:assignvariableop_57_batch_normalization_10_moving_varianceIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp/assignvariableop_58_batch_normalization_8_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp.assignvariableop_59_batch_normalization_8_betaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp5assignvariableop_60_batch_normalization_8_moving_meanIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp9assignvariableop_61_batch_normalization_8_moving_varianceIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp0assignvariableop_62_batch_normalization_11_gammaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp/assignvariableop_63_batch_normalization_11_betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp6assignvariableop_64_batch_normalization_11_moving_meanIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp:assignvariableop_65_batch_normalization_11_moving_varianceIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp#assignvariableop_66_conv2d_9_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp!assignvariableop_67_conv2d_9_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_68Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_69IdentityIdentity_68:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_69Identity_69:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339144

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_4_layer_call_fn_339455

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_336002�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336130

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�D
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338871

inputsB
'conv2d_1_conv2d_readvariableop_resource:�7
(conv2d_1_biasadd_readvariableop_resource:	�<
-batch_normalization_1_readvariableop_resource:	�>
/batch_normalization_1_readvariableop_1_resource:	�M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�@
%conv2d_conv2d_readvariableop_resource:�5
&conv2d_biasadd_readvariableop_resource:	�C
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�:
+batch_normalization_readvariableop_resource:	�<
-batch_normalization_readvariableop_1_resource:	�K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	�M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_3_readvariableop_resource:	�>
/batch_normalization_3_readvariableop_1_resource:	�M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_4_conv2d_readvariableop_resource:��7
(conv2d_4_biasadd_readvariableop_resource:	�<
-batch_normalization_5_readvariableop_resource:	�>
/batch_normalization_5_readvariableop_1_resource:	�M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_3_conv2d_readvariableop_resource:��7
(conv2d_3_biasadd_readvariableop_resource:	�C
'conv2d_5_conv2d_readvariableop_resource:��7
(conv2d_5_biasadd_readvariableop_resource:	�<
-batch_normalization_6_readvariableop_resource:	�>
/batch_normalization_6_readvariableop_1_resource:	�M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_7_readvariableop_resource:	�>
/batch_normalization_7_readvariableop_1_resource:	�M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_7_conv2d_readvariableop_resource:��7
(conv2d_7_biasadd_readvariableop_resource:	�<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_6_conv2d_readvariableop_resource:��7
(conv2d_6_biasadd_readvariableop_resource:	�C
'conv2d_8_conv2d_readvariableop_resource:��7
(conv2d_8_biasadd_readvariableop_resource:	�=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_8_readvariableop_resource:	�>
/batch_normalization_8_readvariableop_1_resource:	�M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�B
'conv2d_9_conv2d_readvariableop_resource:�6
(conv2d_9_biasadd_readvariableop_resource:
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�%batch_normalization_10/AssignNewValue�'batch_normalization_10/AssignNewValue_1�6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�%batch_normalization_11/AssignNewValue�'batch_normalization_11/AssignNewValue_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�$batch_normalization_7/AssignNewValue�&batch_normalization_7/AssignNewValue_1�5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�$batch_normalization_8/AssignNewValue�&batch_normalization_8/AssignNewValue_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
add/addAddV2*batch_normalization_2/FusedBatchNormV3:y:0(batch_normalization/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3add/add:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_4/LeakyRelu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_5/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_5/LeakyRelu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
	add_1/addAddV2*batch_normalization_6/FusedBatchNormV3:y:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3add_1/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_7/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_7/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_7/LeakyRelu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_6/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_8/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_8/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_8/LeakyRelu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_6/LeakyRelu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
	add_2/addAddV2+batch_normalization_10/FusedBatchNormV3:y:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3add_2/add:z:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_9/Conv2DConv2D+batch_normalization_11/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+����������������������������
conv2d_9/SoftmaxSoftmaxconv2d_9/BiasAdd:output:0*
T0*A
_output_shapes/
-:+����������������������������
IdentityIdentityconv2d_9/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
m
A__inference_add_2_layer_call_and_return_conditional_losses_339850
inputs_0
inputs_1
identitym
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,����������������������������j
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_0:lh
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_1
�
�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_336398

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338962

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_336002

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_338900

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336322

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335938

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_336697

inputs*
conv2d_1_336351:�
conv2d_1_336353:	�+
batch_normalization_1_336363:	�+
batch_normalization_1_336365:	�+
batch_normalization_1_336367:	�+
batch_normalization_1_336369:	�(
conv2d_336383:�
conv2d_336385:	�+
conv2d_2_336399:��
conv2d_2_336401:	�+
batch_normalization_2_336418:	�+
batch_normalization_2_336420:	�+
batch_normalization_2_336422:	�+
batch_normalization_2_336424:	�)
batch_normalization_336427:	�)
batch_normalization_336429:	�)
batch_normalization_336431:	�)
batch_normalization_336433:	�+
batch_normalization_3_336444:	�+
batch_normalization_3_336446:	�+
batch_normalization_3_336448:	�+
batch_normalization_3_336450:	�+
conv2d_4_336464:��
conv2d_4_336466:	�+
batch_normalization_5_336476:	�+
batch_normalization_5_336478:	�+
batch_normalization_5_336480:	�+
batch_normalization_5_336482:	�+
conv2d_3_336496:��
conv2d_3_336498:	�+
conv2d_5_336512:��
conv2d_5_336514:	�+
batch_normalization_6_336531:	�+
batch_normalization_6_336533:	�+
batch_normalization_6_336535:	�+
batch_normalization_6_336537:	�+
batch_normalization_4_336540:	�+
batch_normalization_4_336542:	�+
batch_normalization_4_336544:	�+
batch_normalization_4_336546:	�+
batch_normalization_7_336557:	�+
batch_normalization_7_336559:	�+
batch_normalization_7_336561:	�+
batch_normalization_7_336563:	�+
conv2d_7_336577:��
conv2d_7_336579:	�+
batch_normalization_9_336589:	�+
batch_normalization_9_336591:	�+
batch_normalization_9_336593:	�+
batch_normalization_9_336595:	�+
conv2d_6_336609:��
conv2d_6_336611:	�+
conv2d_8_336625:��
conv2d_8_336627:	�,
batch_normalization_10_336644:	�,
batch_normalization_10_336646:	�,
batch_normalization_10_336648:	�,
batch_normalization_10_336650:	�+
batch_normalization_8_336653:	�+
batch_normalization_8_336655:	�+
batch_normalization_8_336657:	�+
batch_normalization_8_336659:	�,
batch_normalization_11_336670:	�,
batch_normalization_11_336672:	�,
batch_normalization_11_336674:	�,
batch_normalization_11_336676:	�*
conv2d_9_336691:�
conv2d_9_336693:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_336351conv2d_1_336353*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_336350�
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_336361�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_336363batch_normalization_1_336365batch_normalization_1_336367batch_normalization_1_336369*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335587�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_336383conv2d_336385*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_336382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_336399conv2d_2_336401*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_336398�
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_336409�
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_336416�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_336418batch_normalization_2_336420batch_normalization_2_336422batch_normalization_2_336424*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335651�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_336427batch_normalization_336429batch_normalization_336431batch_normalization_336433*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335715�
add/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_336442�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_336444batch_normalization_3_336446batch_normalization_3_336448batch_normalization_3_336450*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335779�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_336464conv2d_4_336466*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_336463�
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_336474�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_5_336476batch_normalization_5_336478batch_normalization_5_336480batch_normalization_5_336482*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335843�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_336496conv2d_3_336498*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_336495�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_5_336512conv2d_5_336514*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_336511�
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_336522�
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_336529�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_6_336531batch_normalization_6_336533batch_normalization_6_336535batch_normalization_6_336537*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335907�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_4_336540batch_normalization_4_336542batch_normalization_4_336544batch_normalization_4_336546*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_335971�
add_1/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_336555�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_336557batch_normalization_7_336559batch_normalization_7_336561batch_normalization_7_336563*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336035�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_7_336577conv2d_7_336579*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_336576�
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_336587�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_9_336589batch_normalization_9_336591batch_normalization_9_336593batch_normalization_9_336595*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336099�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_6_336609conv2d_6_336611*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_336608�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_8_336625conv2d_8_336627*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_336624�
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_336635�
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_336642�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0batch_normalization_10_336644batch_normalization_10_336646batch_normalization_10_336648batch_normalization_10_336650*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336163�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_8_336653batch_normalization_8_336655batch_normalization_8_336657batch_normalization_8_336659*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336227�
add_2/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_336668�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_11_336670batch_normalization_11_336672batch_normalization_11_336674batch_normalization_11_336676*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336291�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_9_336691conv2d_9_336693*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_336690�
IdentityIdentity)conv2d_9/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_336350

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335779

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_1_layer_call_fn_338895

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_336361{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336035

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
k
?__inference_add_layer_call_and_return_conditional_losses_339156
inputs_0
inputs_1
identitym
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,����������������������������j
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_0:lh
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_1
�
m
A__inference_add_1_layer_call_and_return_conditional_losses_339503
inputs_0
inputs_1
identitym
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,����������������������������j
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:l h
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_0:lh
B
_output_shapes0
.:,����������������������������
"
_user_specified_name
inputs_1
�
i
?__inference_add_layer_call_and_return_conditional_losses_336442

inputs
inputs_1
identityk
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,����������������������������j
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,����������������������������:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_5_layer_call_fn_339273

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335874�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_339704

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335843

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335907

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339820

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�<
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338626

inputsB
'conv2d_1_conv2d_readvariableop_resource:�7
(conv2d_1_biasadd_readvariableop_resource:	�<
-batch_normalization_1_readvariableop_resource:	�>
/batch_normalization_1_readvariableop_1_resource:	�M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�@
%conv2d_conv2d_readvariableop_resource:�5
&conv2d_biasadd_readvariableop_resource:	�C
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�:
+batch_normalization_readvariableop_resource:	�<
-batch_normalization_readvariableop_1_resource:	�K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	�M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_3_readvariableop_resource:	�>
/batch_normalization_3_readvariableop_1_resource:	�M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_4_conv2d_readvariableop_resource:��7
(conv2d_4_biasadd_readvariableop_resource:	�<
-batch_normalization_5_readvariableop_resource:	�>
/batch_normalization_5_readvariableop_1_resource:	�M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_3_conv2d_readvariableop_resource:��7
(conv2d_3_biasadd_readvariableop_resource:	�C
'conv2d_5_conv2d_readvariableop_resource:��7
(conv2d_5_biasadd_readvariableop_resource:	�<
-batch_normalization_6_readvariableop_resource:	�>
/batch_normalization_6_readvariableop_1_resource:	�M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_7_readvariableop_resource:	�>
/batch_normalization_7_readvariableop_1_resource:	�M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_7_conv2d_readvariableop_resource:��7
(conv2d_7_biasadd_readvariableop_resource:	�<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_6_conv2d_readvariableop_resource:��7
(conv2d_6_biasadd_readvariableop_resource:	�C
'conv2d_8_conv2d_readvariableop_resource:��7
(conv2d_8_biasadd_readvariableop_resource:	�=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�<
-batch_normalization_8_readvariableop_resource:	�>
/batch_normalization_8_readvariableop_1_resource:	�M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�B
'conv2d_9_conv2d_readvariableop_resource:�6
(conv2d_9_biasadd_readvariableop_resource:
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_1/LeakyRelu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_2/LeakyRelu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3#leaky_re_lu/LeakyRelu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
add/addAddV2*batch_normalization_2/FusedBatchNormV3:y:0(batch_normalization/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3add/add:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_4/LeakyRelu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_5/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_3/LeakyRelu	LeakyReluconv2d_3/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
leaky_re_lu_5/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_5/LeakyRelu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_3/LeakyRelu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
	add_1/addAddV2*batch_normalization_6/FusedBatchNormV3:y:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3add_1/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_7/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_7/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_7/LeakyRelu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_6/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_8/Conv2DConv2D*batch_normalization_9/FusedBatchNormV3:y:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_8/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_8/LeakyRelu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%leaky_re_lu_6/LeakyRelu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
	add_2/addAddV2+batch_normalization_10/FusedBatchNormV3:y:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3add_2/add:z:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d_9/Conv2DConv2D+batch_normalization_11/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+����������������������������
conv2d_9/SoftmaxSoftmaxconv2d_9/BiasAdd:output:0*
T0*A
_output_shapes/
-:+����������������������������
IdentityIdentityconv2d_9/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_7_layer_call_fn_339574

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_336576�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_8_layer_call_fn_339699

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_336642{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_11_layer_call_fn_339876

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336322�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_10_layer_call_fn_339727

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336163�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339411

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_8_layer_call_fn_339802

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336258�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
1__inference_ResNet_6channels_layer_call_fn_338381

inputs"
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�$
	unknown_5:�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:	�&

unknown_43:��

unknown_44:	�

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�&

unknown_49:��

unknown_50:	�&

unknown_51:��

unknown_52:	�

unknown_53:	�

unknown_54:	�

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�%

unknown_65:�

unknown_66:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*N
_read_only_resource_inputs0
.,	
 !"%&)*-./0345678;<?@CD*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337328�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_339594

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_8_layer_call_fn_339789

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336227�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339838

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339309

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335715

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_336576

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_5_layer_call_fn_339318

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_336511�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_339932

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������z
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_339247

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_10_layer_call_fn_339740

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336194�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_11_layer_call_fn_339863

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336291�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_7_layer_call_fn_339589

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_336587{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_339714

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_3_layer_call_fn_339169

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335779�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_339675

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339758

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_1_layer_call_fn_338880

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_336350�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339565

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_336463

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_336511

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
H
,__inference_leaky_re_lu_layer_call_fn_339015

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_336409{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335587

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_336522

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_339328

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339473

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_5_layer_call_fn_339352

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_336529{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_layer_call_fn_339095

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335715�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_339367

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_336361

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_3_layer_call_fn_339182

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335810�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_336495

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337956
input_2*
conv2d_1_337785:�
conv2d_1_337787:	�+
batch_normalization_1_337791:	�+
batch_normalization_1_337793:	�+
batch_normalization_1_337795:	�+
batch_normalization_1_337797:	�(
conv2d_337800:�
conv2d_337802:	�+
conv2d_2_337805:��
conv2d_2_337807:	�+
batch_normalization_2_337812:	�+
batch_normalization_2_337814:	�+
batch_normalization_2_337816:	�+
batch_normalization_2_337818:	�)
batch_normalization_337821:	�)
batch_normalization_337823:	�)
batch_normalization_337825:	�)
batch_normalization_337827:	�+
batch_normalization_3_337831:	�+
batch_normalization_3_337833:	�+
batch_normalization_3_337835:	�+
batch_normalization_3_337837:	�+
conv2d_4_337840:��
conv2d_4_337842:	�+
batch_normalization_5_337846:	�+
batch_normalization_5_337848:	�+
batch_normalization_5_337850:	�+
batch_normalization_5_337852:	�+
conv2d_3_337855:��
conv2d_3_337857:	�+
conv2d_5_337860:��
conv2d_5_337862:	�+
batch_normalization_6_337867:	�+
batch_normalization_6_337869:	�+
batch_normalization_6_337871:	�+
batch_normalization_6_337873:	�+
batch_normalization_4_337876:	�+
batch_normalization_4_337878:	�+
batch_normalization_4_337880:	�+
batch_normalization_4_337882:	�+
batch_normalization_7_337886:	�+
batch_normalization_7_337888:	�+
batch_normalization_7_337890:	�+
batch_normalization_7_337892:	�+
conv2d_7_337895:��
conv2d_7_337897:	�+
batch_normalization_9_337901:	�+
batch_normalization_9_337903:	�+
batch_normalization_9_337905:	�+
batch_normalization_9_337907:	�+
conv2d_6_337910:��
conv2d_6_337912:	�+
conv2d_8_337915:��
conv2d_8_337917:	�,
batch_normalization_10_337922:	�,
batch_normalization_10_337924:	�,
batch_normalization_10_337926:	�,
batch_normalization_10_337928:	�+
batch_normalization_8_337931:	�+
batch_normalization_8_337933:	�+
batch_normalization_8_337935:	�+
batch_normalization_8_337937:	�,
batch_normalization_11_337941:	�,
batch_normalization_11_337943:	�,
batch_normalization_11_337945:	�,
batch_normalization_11_337947:	�*
conv2d_9_337950:�
conv2d_9_337952:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1_337785conv2d_1_337787*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_336350�
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_336361�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_337791batch_normalization_1_337793batch_normalization_1_337795batch_normalization_1_337797*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335618�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_337800conv2d_337802*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_336382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_337805conv2d_2_337807*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_336398�
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_336409�
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_336416�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_337812batch_normalization_2_337814batch_normalization_2_337816batch_normalization_2_337818*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335682�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_337821batch_normalization_337823batch_normalization_337825batch_normalization_337827*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335746�
add/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_336442�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_337831batch_normalization_3_337833batch_normalization_3_337835batch_normalization_3_337837*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335810�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_337840conv2d_4_337842*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_336463�
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_336474�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_5_337846batch_normalization_5_337848batch_normalization_5_337850batch_normalization_5_337852*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335874�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_337855conv2d_3_337857*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_336495�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_5_337860conv2d_5_337862*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_336511�
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_336522�
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_336529�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_6_337867batch_normalization_6_337869batch_normalization_6_337871batch_normalization_6_337873*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335938�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_4_337876batch_normalization_4_337878batch_normalization_4_337880batch_normalization_4_337882*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_336002�
add_1/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_336555�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_337886batch_normalization_7_337888batch_normalization_7_337890batch_normalization_7_337892*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336066�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_7_337895conv2d_7_337897*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_336576�
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_336587�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_9_337901batch_normalization_9_337903batch_normalization_9_337905batch_normalization_9_337907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336130�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_6_337910conv2d_6_337912*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_336608�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_8_337915conv2d_8_337917*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_336624�
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_336635�
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_336642�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0batch_normalization_10_337922batch_normalization_10_337924batch_normalization_10_337926batch_normalization_10_337928*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336194�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_8_337931batch_normalization_8_337933batch_normalization_8_337935batch_normalization_8_337937*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336258�
add_2/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_336668�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_11_337941batch_normalization_11_337943batch_normalization_11_337945batch_normalization_11_337947*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336322�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_9_337950conv2d_9_337952*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_336690�
IdentityIdentity)conv2d_9/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335651

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339082

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337782
input_2*
conv2d_1_337611:�
conv2d_1_337613:	�+
batch_normalization_1_337617:	�+
batch_normalization_1_337619:	�+
batch_normalization_1_337621:	�+
batch_normalization_1_337623:	�(
conv2d_337626:�
conv2d_337628:	�+
conv2d_2_337631:��
conv2d_2_337633:	�+
batch_normalization_2_337638:	�+
batch_normalization_2_337640:	�+
batch_normalization_2_337642:	�+
batch_normalization_2_337644:	�)
batch_normalization_337647:	�)
batch_normalization_337649:	�)
batch_normalization_337651:	�)
batch_normalization_337653:	�+
batch_normalization_3_337657:	�+
batch_normalization_3_337659:	�+
batch_normalization_3_337661:	�+
batch_normalization_3_337663:	�+
conv2d_4_337666:��
conv2d_4_337668:	�+
batch_normalization_5_337672:	�+
batch_normalization_5_337674:	�+
batch_normalization_5_337676:	�+
batch_normalization_5_337678:	�+
conv2d_3_337681:��
conv2d_3_337683:	�+
conv2d_5_337686:��
conv2d_5_337688:	�+
batch_normalization_6_337693:	�+
batch_normalization_6_337695:	�+
batch_normalization_6_337697:	�+
batch_normalization_6_337699:	�+
batch_normalization_4_337702:	�+
batch_normalization_4_337704:	�+
batch_normalization_4_337706:	�+
batch_normalization_4_337708:	�+
batch_normalization_7_337712:	�+
batch_normalization_7_337714:	�+
batch_normalization_7_337716:	�+
batch_normalization_7_337718:	�+
conv2d_7_337721:��
conv2d_7_337723:	�+
batch_normalization_9_337727:	�+
batch_normalization_9_337729:	�+
batch_normalization_9_337731:	�+
batch_normalization_9_337733:	�+
conv2d_6_337736:��
conv2d_6_337738:	�+
conv2d_8_337741:��
conv2d_8_337743:	�,
batch_normalization_10_337748:	�,
batch_normalization_10_337750:	�,
batch_normalization_10_337752:	�,
batch_normalization_10_337754:	�+
batch_normalization_8_337757:	�+
batch_normalization_8_337759:	�+
batch_normalization_8_337761:	�+
batch_normalization_8_337763:	�,
batch_normalization_11_337767:	�,
batch_normalization_11_337769:	�,
batch_normalization_11_337771:	�,
batch_normalization_11_337773:	�*
conv2d_9_337776:�
conv2d_9_337778:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1_337611conv2d_1_337613*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_336350�
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_336361�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_337617batch_normalization_1_337619batch_normalization_1_337621batch_normalization_1_337623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335587�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_337626conv2d_337628*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_336382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_337631conv2d_2_337633*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_336398�
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_336409�
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_336416�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_337638batch_normalization_2_337640batch_normalization_2_337642batch_normalization_2_337644*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335651�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_337647batch_normalization_337649batch_normalization_337651batch_normalization_337653*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335715�
add/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_336442�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_337657batch_normalization_3_337659batch_normalization_3_337661batch_normalization_3_337663*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335779�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_337666conv2d_4_337668*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_336463�
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_336474�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_5_337672batch_normalization_5_337674batch_normalization_5_337676batch_normalization_5_337678*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335843�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_337681conv2d_3_337683*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_336495�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_5_337686conv2d_5_337688*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_336511�
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_336522�
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_336529�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_6_337693batch_normalization_6_337695batch_normalization_6_337697batch_normalization_6_337699*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335907�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_4_337702batch_normalization_4_337704batch_normalization_4_337706batch_normalization_4_337708*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_335971�
add_1/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_336555�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_337712batch_normalization_7_337714batch_normalization_7_337716batch_normalization_7_337718*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336035�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_7_337721conv2d_7_337723*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_336576�
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_336587�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_9_337727batch_normalization_9_337729batch_normalization_9_337731batch_normalization_9_337733*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336099�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_6_337736conv2d_6_337738*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_336608�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_8_337741conv2d_8_337743*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_336624�
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_336635�
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_336642�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0batch_normalization_10_337748batch_normalization_10_337750batch_normalization_10_337752batch_normalization_10_337754*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336163�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_8_337757batch_normalization_8_337759batch_normalization_8_337761batch_normalization_8_337763*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336227�
add_2/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_336668�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_11_337767batch_normalization_11_337769batch_normalization_11_337771batch_normalization_11_337773*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336291�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_9_337776conv2d_9_337778*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_336690�
IdentityIdentity)conv2d_9/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2
�
e
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_336416

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339912

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339291

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337328

inputs*
conv2d_1_337157:�
conv2d_1_337159:	�+
batch_normalization_1_337163:	�+
batch_normalization_1_337165:	�+
batch_normalization_1_337167:	�+
batch_normalization_1_337169:	�(
conv2d_337172:�
conv2d_337174:	�+
conv2d_2_337177:��
conv2d_2_337179:	�+
batch_normalization_2_337184:	�+
batch_normalization_2_337186:	�+
batch_normalization_2_337188:	�+
batch_normalization_2_337190:	�)
batch_normalization_337193:	�)
batch_normalization_337195:	�)
batch_normalization_337197:	�)
batch_normalization_337199:	�+
batch_normalization_3_337203:	�+
batch_normalization_3_337205:	�+
batch_normalization_3_337207:	�+
batch_normalization_3_337209:	�+
conv2d_4_337212:��
conv2d_4_337214:	�+
batch_normalization_5_337218:	�+
batch_normalization_5_337220:	�+
batch_normalization_5_337222:	�+
batch_normalization_5_337224:	�+
conv2d_3_337227:��
conv2d_3_337229:	�+
conv2d_5_337232:��
conv2d_5_337234:	�+
batch_normalization_6_337239:	�+
batch_normalization_6_337241:	�+
batch_normalization_6_337243:	�+
batch_normalization_6_337245:	�+
batch_normalization_4_337248:	�+
batch_normalization_4_337250:	�+
batch_normalization_4_337252:	�+
batch_normalization_4_337254:	�+
batch_normalization_7_337258:	�+
batch_normalization_7_337260:	�+
batch_normalization_7_337262:	�+
batch_normalization_7_337264:	�+
conv2d_7_337267:��
conv2d_7_337269:	�+
batch_normalization_9_337273:	�+
batch_normalization_9_337275:	�+
batch_normalization_9_337277:	�+
batch_normalization_9_337279:	�+
conv2d_6_337282:��
conv2d_6_337284:	�+
conv2d_8_337287:��
conv2d_8_337289:	�,
batch_normalization_10_337294:	�,
batch_normalization_10_337296:	�,
batch_normalization_10_337298:	�,
batch_normalization_10_337300:	�+
batch_normalization_8_337303:	�+
batch_normalization_8_337305:	�+
batch_normalization_8_337307:	�+
batch_normalization_8_337309:	�,
batch_normalization_11_337313:	�,
batch_normalization_11_337315:	�,
batch_normalization_11_337317:	�,
batch_normalization_11_337319:	�*
conv2d_9_337322:�
conv2d_9_337324:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall� conv2d_5/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_337157conv2d_1_337159*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_336350�
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_336361�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0batch_normalization_1_337163batch_normalization_1_337165batch_normalization_1_337167batch_normalization_1_337169*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335618�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_337172conv2d_337174*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_336382�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_337177conv2d_2_337179*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_336398�
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_336409�
leaky_re_lu_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_336416�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0batch_normalization_2_337184batch_normalization_2_337186batch_normalization_2_337188batch_normalization_2_337190*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_335682�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0batch_normalization_337193batch_normalization_337195batch_normalization_337197batch_normalization_337199*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_335746�
add/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:04batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_336442�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_337203batch_normalization_3_337205batch_normalization_3_337207batch_normalization_3_337209*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_335810�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_337212conv2d_4_337214*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_336463�
leaky_re_lu_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_336474�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0batch_normalization_5_337218batch_normalization_5_337220batch_normalization_5_337222batch_normalization_5_337224*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_335874�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_337227conv2d_3_337229*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_336495�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_5_337232conv2d_5_337234*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_336511�
leaky_re_lu_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_336522�
leaky_re_lu_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_336529�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0batch_normalization_6_337239batch_normalization_6_337241batch_normalization_6_337243batch_normalization_6_337245*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_335938�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0batch_normalization_4_337248batch_normalization_4_337250batch_normalization_4_337252batch_normalization_4_337254*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_336002�
add_1/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_336555�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_337258batch_normalization_7_337260batch_normalization_7_337262batch_normalization_7_337264*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336066�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_7_337267conv2d_7_337269*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_336576�
leaky_re_lu_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_336587�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0batch_normalization_9_337273batch_normalization_9_337275batch_normalization_9_337277batch_normalization_9_337279*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336130�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_6_337282conv2d_6_337284*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_336608�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0conv2d_8_337287conv2d_8_337289*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_336624�
leaky_re_lu_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_336635�
leaky_re_lu_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_336642�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_8/PartitionedCall:output:0batch_normalization_10_337294batch_normalization_10_337296batch_normalization_10_337298batch_normalization_10_337300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336194�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0batch_normalization_8_337303batch_normalization_8_337305batch_normalization_8_337307batch_normalization_8_337309*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336258�
add_2/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_336668�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_11_337313batch_normalization_11_337315batch_normalization_11_337317batch_normalization_11_337319*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_336322�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0conv2d_9_337322conv2d_9_337324*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_336690�
IdentityIdentity)conv2d_9/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_4_layer_call_fn_339227

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_336463�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
ފ
�N
!__inference__wrapped_model_335565
input_2S
8resnet_6channels_conv2d_1_conv2d_readvariableop_resource:�H
9resnet_6channels_conv2d_1_biasadd_readvariableop_resource:	�M
>resnet_6channels_batch_normalization_1_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_1_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�Q
6resnet_6channels_conv2d_conv2d_readvariableop_resource:�F
7resnet_6channels_conv2d_biasadd_readvariableop_resource:	�T
8resnet_6channels_conv2d_2_conv2d_readvariableop_resource:��H
9resnet_6channels_conv2d_2_biasadd_readvariableop_resource:	�M
>resnet_6channels_batch_normalization_2_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_2_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�K
<resnet_6channels_batch_normalization_readvariableop_resource:	�M
>resnet_6channels_batch_normalization_readvariableop_1_resource:	�\
Mresnet_6channels_batch_normalization_fusedbatchnormv3_readvariableop_resource:	�^
Oresnet_6channels_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	�M
>resnet_6channels_batch_normalization_3_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_3_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�T
8resnet_6channels_conv2d_4_conv2d_readvariableop_resource:��H
9resnet_6channels_conv2d_4_biasadd_readvariableop_resource:	�M
>resnet_6channels_batch_normalization_5_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_5_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�T
8resnet_6channels_conv2d_3_conv2d_readvariableop_resource:��H
9resnet_6channels_conv2d_3_biasadd_readvariableop_resource:	�T
8resnet_6channels_conv2d_5_conv2d_readvariableop_resource:��H
9resnet_6channels_conv2d_5_biasadd_readvariableop_resource:	�M
>resnet_6channels_batch_normalization_6_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_6_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�M
>resnet_6channels_batch_normalization_4_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_4_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�M
>resnet_6channels_batch_normalization_7_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_7_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�T
8resnet_6channels_conv2d_7_conv2d_readvariableop_resource:��H
9resnet_6channels_conv2d_7_biasadd_readvariableop_resource:	�M
>resnet_6channels_batch_normalization_9_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_9_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�T
8resnet_6channels_conv2d_6_conv2d_readvariableop_resource:��H
9resnet_6channels_conv2d_6_biasadd_readvariableop_resource:	�T
8resnet_6channels_conv2d_8_conv2d_readvariableop_resource:��H
9resnet_6channels_conv2d_8_biasadd_readvariableop_resource:	�N
?resnet_6channels_batch_normalization_10_readvariableop_resource:	�P
Aresnet_6channels_batch_normalization_10_readvariableop_1_resource:	�_
Presnet_6channels_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�a
Rresnet_6channels_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�M
>resnet_6channels_batch_normalization_8_readvariableop_resource:	�O
@resnet_6channels_batch_normalization_8_readvariableop_1_resource:	�^
Oresnet_6channels_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�`
Qresnet_6channels_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�N
?resnet_6channels_batch_normalization_11_readvariableop_resource:	�P
Aresnet_6channels_batch_normalization_11_readvariableop_1_resource:	�_
Presnet_6channels_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�a
Rresnet_6channels_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�S
8resnet_6channels_conv2d_9_conv2d_readvariableop_resource:�G
9resnet_6channels_conv2d_9_biasadd_readvariableop_resource:
identity��DResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp�FResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�3ResNet_6channels/batch_normalization/ReadVariableOp�5ResNet_6channels/batch_normalization/ReadVariableOp_1�FResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_1/ReadVariableOp�7ResNet_6channels/batch_normalization_1/ReadVariableOp_1�GResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp�IResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�6ResNet_6channels/batch_normalization_10/ReadVariableOp�8ResNet_6channels/batch_normalization_10/ReadVariableOp_1�GResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp�IResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�6ResNet_6channels/batch_normalization_11/ReadVariableOp�8ResNet_6channels/batch_normalization_11/ReadVariableOp_1�FResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_2/ReadVariableOp�7ResNet_6channels/batch_normalization_2/ReadVariableOp_1�FResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_3/ReadVariableOp�7ResNet_6channels/batch_normalization_3/ReadVariableOp_1�FResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_4/ReadVariableOp�7ResNet_6channels/batch_normalization_4/ReadVariableOp_1�FResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_5/ReadVariableOp�7ResNet_6channels/batch_normalization_5/ReadVariableOp_1�FResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_6/ReadVariableOp�7ResNet_6channels/batch_normalization_6/ReadVariableOp_1�FResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_7/ReadVariableOp�7ResNet_6channels/batch_normalization_7/ReadVariableOp_1�FResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_8/ReadVariableOp�7ResNet_6channels/batch_normalization_8/ReadVariableOp_1�FResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�HResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�5ResNet_6channels/batch_normalization_9/ReadVariableOp�7ResNet_6channels/batch_normalization_9/ReadVariableOp_1�.ResNet_6channels/conv2d/BiasAdd/ReadVariableOp�-ResNet_6channels/conv2d/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_1/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_1/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_2/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_2/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_3/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_3/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_4/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_4/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_5/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_5/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_6/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_6/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_7/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_7/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_8/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_8/Conv2D/ReadVariableOp�0ResNet_6channels/conv2d_9/BiasAdd/ReadVariableOp�/ResNet_6channels/conv2d_9/Conv2D/ReadVariableOp�
/ResNet_6channels/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
 ResNet_6channels/conv2d_1/Conv2DConv2Dinput_27ResNet_6channels/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_1/BiasAddBiasAdd)ResNet_6channels/conv2d_1/Conv2D:output:08ResNet_6channels/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
(ResNet_6channels/leaky_re_lu_1/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_1/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
5ResNet_6channels/batch_normalization_1/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_1/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_1/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_1/LeakyRelu:activations:0=ResNet_6channels/batch_normalization_1/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_1/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
-ResNet_6channels/conv2d/Conv2D/ReadVariableOpReadVariableOp6resnet_6channels_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
ResNet_6channels/conv2d/Conv2DConv2Dinput_25ResNet_6channels/conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
.ResNet_6channels/conv2d/BiasAdd/ReadVariableOpReadVariableOp7resnet_6channels_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ResNet_6channels/conv2d/BiasAddBiasAdd'ResNet_6channels/conv2d/Conv2D:output:06ResNet_6channels/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
/ResNet_6channels/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 ResNet_6channels/conv2d_2/Conv2DConv2D;ResNet_6channels/batch_normalization_1/FusedBatchNormV3:y:07ResNet_6channels/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_2/BiasAddBiasAdd)ResNet_6channels/conv2d_2/Conv2D:output:08ResNet_6channels/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
&ResNet_6channels/leaky_re_lu/LeakyRelu	LeakyRelu(ResNet_6channels/conv2d/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
(ResNet_6channels/leaky_re_lu_2/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_2/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
5ResNet_6channels/batch_normalization_2/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_2/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_2/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_2/LeakyRelu:activations:0=ResNet_6channels/batch_normalization_2/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_2/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
3ResNet_6channels/batch_normalization/ReadVariableOpReadVariableOp<resnet_6channels_batch_normalization_readvariableop_resource*
_output_shapes	
:�*
dtype0�
5ResNet_6channels/batch_normalization/ReadVariableOp_1ReadVariableOp>resnet_6channels_batch_normalization_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
DResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMresnet_6channels_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOresnet_6channels_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5ResNet_6channels/batch_normalization/FusedBatchNormV3FusedBatchNormV34ResNet_6channels/leaky_re_lu/LeakyRelu:activations:0;ResNet_6channels/batch_normalization/ReadVariableOp:value:0=ResNet_6channels/batch_normalization/ReadVariableOp_1:value:0LResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0NResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
ResNet_6channels/add/addAddV2;ResNet_6channels/batch_normalization_2/FusedBatchNormV3:y:09ResNet_6channels/batch_normalization/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
5ResNet_6channels/batch_normalization_3/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_3/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3ResNet_6channels/add/add:z:0=ResNet_6channels/batch_normalization_3/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_3/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
/ResNet_6channels/conv2d_4/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 ResNet_6channels/conv2d_4/Conv2DConv2D;ResNet_6channels/batch_normalization_3/FusedBatchNormV3:y:07ResNet_6channels/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_4/BiasAddBiasAdd)ResNet_6channels/conv2d_4/Conv2D:output:08ResNet_6channels/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
(ResNet_6channels/leaky_re_lu_4/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_4/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
5ResNet_6channels/batch_normalization_5/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_5/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_5/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_4/LeakyRelu:activations:0=ResNet_6channels/batch_normalization_5/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_5/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
/ResNet_6channels/conv2d_3/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 ResNet_6channels/conv2d_3/Conv2DConv2D;ResNet_6channels/batch_normalization_3/FusedBatchNormV3:y:07ResNet_6channels/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_3/BiasAddBiasAdd)ResNet_6channels/conv2d_3/Conv2D:output:08ResNet_6channels/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
/ResNet_6channels/conv2d_5/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 ResNet_6channels/conv2d_5/Conv2DConv2D;ResNet_6channels/batch_normalization_5/FusedBatchNormV3:y:07ResNet_6channels/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_5/BiasAddBiasAdd)ResNet_6channels/conv2d_5/Conv2D:output:08ResNet_6channels/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
(ResNet_6channels/leaky_re_lu_3/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_3/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
(ResNet_6channels/leaky_re_lu_5/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_5/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
5ResNet_6channels/batch_normalization_6/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_6/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_6/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_5/LeakyRelu:activations:0=ResNet_6channels/batch_normalization_6/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_6/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
5ResNet_6channels/batch_normalization_4/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_4/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_4/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_3/LeakyRelu:activations:0=ResNet_6channels/batch_normalization_4/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_4/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
ResNet_6channels/add_1/addAddV2;ResNet_6channels/batch_normalization_6/FusedBatchNormV3:y:0;ResNet_6channels/batch_normalization_4/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
5ResNet_6channels/batch_normalization_7/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_7/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3ResNet_6channels/add_1/add:z:0=ResNet_6channels/batch_normalization_7/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_7/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
/ResNet_6channels/conv2d_7/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 ResNet_6channels/conv2d_7/Conv2DConv2D;ResNet_6channels/batch_normalization_7/FusedBatchNormV3:y:07ResNet_6channels/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_7/BiasAddBiasAdd)ResNet_6channels/conv2d_7/Conv2D:output:08ResNet_6channels/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
(ResNet_6channels/leaky_re_lu_7/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_7/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
5ResNet_6channels/batch_normalization_9/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_9/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_9/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_7/LeakyRelu:activations:0=ResNet_6channels/batch_normalization_9/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_9/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
/ResNet_6channels/conv2d_6/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 ResNet_6channels/conv2d_6/Conv2DConv2D;ResNet_6channels/batch_normalization_7/FusedBatchNormV3:y:07ResNet_6channels/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_6/BiasAddBiasAdd)ResNet_6channels/conv2d_6/Conv2D:output:08ResNet_6channels/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
/ResNet_6channels/conv2d_8/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 ResNet_6channels/conv2d_8/Conv2DConv2D;ResNet_6channels/batch_normalization_9/FusedBatchNormV3:y:07ResNet_6channels/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!ResNet_6channels/conv2d_8/BiasAddBiasAdd)ResNet_6channels/conv2d_8/Conv2D:output:08ResNet_6channels/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
(ResNet_6channels/leaky_re_lu_6/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_6/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
(ResNet_6channels/leaky_re_lu_8/LeakyRelu	LeakyRelu*ResNet_6channels/conv2d_8/BiasAdd:output:0*B
_output_shapes0
.:,����������������������������*
alpha%���=�
6ResNet_6channels/batch_normalization_10/ReadVariableOpReadVariableOp?resnet_6channels_batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8ResNet_6channels/batch_normalization_10/ReadVariableOp_1ReadVariableOpAresnet_6channels_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
GResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpPresnet_6channels_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
IResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRresnet_6channels_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
8ResNet_6channels/batch_normalization_10/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_8/LeakyRelu:activations:0>ResNet_6channels/batch_normalization_10/ReadVariableOp:value:0@ResNet_6channels/batch_normalization_10/ReadVariableOp_1:value:0OResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0QResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
5ResNet_6channels/batch_normalization_8/ReadVariableOpReadVariableOp>resnet_6channels_batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_8/ReadVariableOp_1ReadVariableOp@resnet_6channels_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpOresnet_6channels_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
HResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresnet_6channels_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7ResNet_6channels/batch_normalization_8/FusedBatchNormV3FusedBatchNormV36ResNet_6channels/leaky_re_lu_6/LeakyRelu:activations:0=ResNet_6channels/batch_normalization_8/ReadVariableOp:value:0?ResNet_6channels/batch_normalization_8/ReadVariableOp_1:value:0NResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0PResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
ResNet_6channels/add_2/addAddV2<ResNet_6channels/batch_normalization_10/FusedBatchNormV3:y:0;ResNet_6channels/batch_normalization_8/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,�����������������������������
6ResNet_6channels/batch_normalization_11/ReadVariableOpReadVariableOp?resnet_6channels_batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8ResNet_6channels/batch_normalization_11/ReadVariableOp_1ReadVariableOpAresnet_6channels_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
GResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpPresnet_6channels_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
IResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRresnet_6channels_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
8ResNet_6channels/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3ResNet_6channels/add_2/add:z:0>ResNet_6channels/batch_normalization_11/ReadVariableOp:value:0@ResNet_6channels/batch_normalization_11/ReadVariableOp_1:value:0OResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0QResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
/ResNet_6channels/conv2d_9/Conv2D/ReadVariableOpReadVariableOp8resnet_6channels_conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
 ResNet_6channels/conv2d_9/Conv2DConv2D<ResNet_6channels/batch_normalization_11/FusedBatchNormV3:y:07ResNet_6channels/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
�
0ResNet_6channels/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp9resnet_6channels_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!ResNet_6channels/conv2d_9/BiasAddBiasAdd)ResNet_6channels/conv2d_9/Conv2D:output:08ResNet_6channels/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+����������������������������
!ResNet_6channels/conv2d_9/SoftmaxSoftmax*ResNet_6channels/conv2d_9/BiasAdd:output:0*
T0*A
_output_shapes/
-:+����������������������������
IdentityIdentity+ResNet_6channels/conv2d_9/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������� 
NoOpNoOpE^ResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOpG^ResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^ResNet_6channels/batch_normalization/ReadVariableOp6^ResNet_6channels/batch_normalization/ReadVariableOp_1G^ResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_1/ReadVariableOp8^ResNet_6channels/batch_normalization_1/ReadVariableOp_1H^ResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOpJ^ResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_17^ResNet_6channels/batch_normalization_10/ReadVariableOp9^ResNet_6channels/batch_normalization_10/ReadVariableOp_1H^ResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOpJ^ResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_17^ResNet_6channels/batch_normalization_11/ReadVariableOp9^ResNet_6channels/batch_normalization_11/ReadVariableOp_1G^ResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_2/ReadVariableOp8^ResNet_6channels/batch_normalization_2/ReadVariableOp_1G^ResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_3/ReadVariableOp8^ResNet_6channels/batch_normalization_3/ReadVariableOp_1G^ResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_4/ReadVariableOp8^ResNet_6channels/batch_normalization_4/ReadVariableOp_1G^ResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_5/ReadVariableOp8^ResNet_6channels/batch_normalization_5/ReadVariableOp_1G^ResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_6/ReadVariableOp8^ResNet_6channels/batch_normalization_6/ReadVariableOp_1G^ResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_7/ReadVariableOp8^ResNet_6channels/batch_normalization_7/ReadVariableOp_1G^ResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_8/ReadVariableOp8^ResNet_6channels/batch_normalization_8/ReadVariableOp_1G^ResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOpI^ResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_16^ResNet_6channels/batch_normalization_9/ReadVariableOp8^ResNet_6channels/batch_normalization_9/ReadVariableOp_1/^ResNet_6channels/conv2d/BiasAdd/ReadVariableOp.^ResNet_6channels/conv2d/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_1/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_1/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_2/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_2/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_3/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_3/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_4/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_4/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_5/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_5/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_6/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_6/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_7/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_7/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_8/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_8/Conv2D/ReadVariableOp1^ResNet_6channels/conv2d_9/BiasAdd/ReadVariableOp0^ResNet_6channels/conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2�
DResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOpDResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
FResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp_1FResNet_6channels/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3ResNet_6channels/batch_normalization/ReadVariableOp3ResNet_6channels/batch_normalization/ReadVariableOp2n
5ResNet_6channels/batch_normalization/ReadVariableOp_15ResNet_6channels/batch_normalization/ReadVariableOp_12�
FResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_1/ReadVariableOp5ResNet_6channels/batch_normalization_1/ReadVariableOp2r
7ResNet_6channels/batch_normalization_1/ReadVariableOp_17ResNet_6channels/batch_normalization_1/ReadVariableOp_12�
GResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOpGResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2�
IResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1IResNet_6channels/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12p
6ResNet_6channels/batch_normalization_10/ReadVariableOp6ResNet_6channels/batch_normalization_10/ReadVariableOp2t
8ResNet_6channels/batch_normalization_10/ReadVariableOp_18ResNet_6channels/batch_normalization_10/ReadVariableOp_12�
GResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOpGResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2�
IResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1IResNet_6channels/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12p
6ResNet_6channels/batch_normalization_11/ReadVariableOp6ResNet_6channels/batch_normalization_11/ReadVariableOp2t
8ResNet_6channels/batch_normalization_11/ReadVariableOp_18ResNet_6channels/batch_normalization_11/ReadVariableOp_12�
FResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_2/ReadVariableOp5ResNet_6channels/batch_normalization_2/ReadVariableOp2r
7ResNet_6channels/batch_normalization_2/ReadVariableOp_17ResNet_6channels/batch_normalization_2/ReadVariableOp_12�
FResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_3/ReadVariableOp5ResNet_6channels/batch_normalization_3/ReadVariableOp2r
7ResNet_6channels/batch_normalization_3/ReadVariableOp_17ResNet_6channels/batch_normalization_3/ReadVariableOp_12�
FResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_4/ReadVariableOp5ResNet_6channels/batch_normalization_4/ReadVariableOp2r
7ResNet_6channels/batch_normalization_4/ReadVariableOp_17ResNet_6channels/batch_normalization_4/ReadVariableOp_12�
FResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_5/ReadVariableOp5ResNet_6channels/batch_normalization_5/ReadVariableOp2r
7ResNet_6channels/batch_normalization_5/ReadVariableOp_17ResNet_6channels/batch_normalization_5/ReadVariableOp_12�
FResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_6/ReadVariableOp5ResNet_6channels/batch_normalization_6/ReadVariableOp2r
7ResNet_6channels/batch_normalization_6/ReadVariableOp_17ResNet_6channels/batch_normalization_6/ReadVariableOp_12�
FResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_7/ReadVariableOp5ResNet_6channels/batch_normalization_7/ReadVariableOp2r
7ResNet_6channels/batch_normalization_7/ReadVariableOp_17ResNet_6channels/batch_normalization_7/ReadVariableOp_12�
FResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_8/ReadVariableOp5ResNet_6channels/batch_normalization_8/ReadVariableOp2r
7ResNet_6channels/batch_normalization_8/ReadVariableOp_17ResNet_6channels/batch_normalization_8/ReadVariableOp_12�
FResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOpFResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
HResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1HResNet_6channels/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12n
5ResNet_6channels/batch_normalization_9/ReadVariableOp5ResNet_6channels/batch_normalization_9/ReadVariableOp2r
7ResNet_6channels/batch_normalization_9/ReadVariableOp_17ResNet_6channels/batch_normalization_9/ReadVariableOp_12`
.ResNet_6channels/conv2d/BiasAdd/ReadVariableOp.ResNet_6channels/conv2d/BiasAdd/ReadVariableOp2^
-ResNet_6channels/conv2d/Conv2D/ReadVariableOp-ResNet_6channels/conv2d/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_1/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_1/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_1/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_1/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_2/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_2/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_2/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_2/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_3/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_3/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_3/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_3/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_4/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_4/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_4/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_4/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_5/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_5/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_5/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_5/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_6/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_6/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_6/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_6/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_7/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_7/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_7/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_7/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_8/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_8/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_8/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_8/Conv2D/ReadVariableOp2d
0ResNet_6channels/conv2d_9/BiasAdd/ReadVariableOp0ResNet_6channels/conv2d_9/BiasAdd/ReadVariableOp2b
/ResNet_6channels/conv2d_9/Conv2D/ReadVariableOp/ResNet_6channels/conv2d_9/Conv2D/ReadVariableOp:j f
A
_output_shapes/
-:+���������������������������
!
_user_specified_name	input_2
�
�
)__inference_conv2d_3_layer_call_fn_339337

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_336495�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_1_layer_call_fn_338913

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335587�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338944

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_4_layer_call_fn_339442

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_335971�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
J
.__inference_leaky_re_lu_4_layer_call_fn_339242

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_336474{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_9_layer_call_fn_339921

inputs"
unknown:�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_336690�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339547

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_7_layer_call_fn_339529

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_336066�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
1__inference_ResNet_6channels_layer_call_fn_338240

inputs"
unknown:�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�$
	unknown_5:�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�&

unknown_21:��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�&

unknown_27:��

unknown_28:	�&

unknown_29:��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:	�&

unknown_43:��

unknown_44:	�

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�&

unknown_49:��

unknown_50:	�&

unknown_51:��

unknown_52:	�

unknown_53:	�

unknown_54:	�

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�%

unknown_65:�

unknown_66:
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_336697�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:+���������������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
6__inference_batch_normalization_1_layer_call_fn_338926

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_335618�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339656

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
B__inference_conv2d_layer_call_and_return_conditional_losses_336382

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_336099

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_336690

inputs9
conv2d_readvariableop_resource:�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������z
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_336258

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_6_layer_call_fn_339684

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_336608�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_336163

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
� 
__inference__traced_save_340159
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: � 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*�
value�B�EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *S
dtypesI
G2E�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:�:�:�:�:��:�:�:�:�:�:�:�:�:�:�:�:�:�:�:�:��:�:�:�:�:�:��:�:��:�:�:�:�:�:�:�:�:�:�:�:�:�:��:�:�:�:�:�:��:�:��:�:�:�:�:�:�:�:�:�:�:�:�:�:�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-	)
'
_output_shapes
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:! 

_output_shapes	
:�:!!

_output_shapes	
:�:!"

_output_shapes	
:�:!#

_output_shapes	
:�:!$

_output_shapes	
:�:!%

_output_shapes	
:�:!&

_output_shapes	
:�:!'

_output_shapes	
:�:!(

_output_shapes	
:�:!)

_output_shapes	
:�:!*

_output_shapes	
:�:!+

_output_shapes	
:�:!,

_output_shapes	
:�:.-*
(
_output_shapes
:��:!.

_output_shapes	
:�:!/

_output_shapes	
:�:!0

_output_shapes	
:�:!1

_output_shapes	
:�:!2

_output_shapes	
:�:.3*
(
_output_shapes
:��:!4

_output_shapes	
:�:.5*
(
_output_shapes
:��:!6

_output_shapes	
:�:!7

_output_shapes	
:�:!8

_output_shapes	
:�:!9

_output_shapes	
:�:!:

_output_shapes	
:�:!;

_output_shapes	
:�:!<

_output_shapes	
:�:!=

_output_shapes	
:�:!>

_output_shapes	
:�:!?

_output_shapes	
:�:!@

_output_shapes	
:�:!A

_output_shapes	
:�:!B

_output_shapes	
:�:-C)
'
_output_shapes
:�: D

_output_shapes
::E

_output_shapes
: 
�
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_336529

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,����������������������������*
alpha%���=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,����������������������������:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339429

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
input_2J
serving_default_input_2:0+���������������������������V
conv2d_9J
StatefulPartitionedCall:0+���������������������������tensorflow/serving/predict:��
�	
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer_with_weights-13
layer-22
layer_with_weights-14
layer-23
layer-24
layer_with_weights-15
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer_with_weights-18
layer-30
 layer_with_weights-19
 layer-31
!layer-32
"layer_with_weights-20
"layer-33
#layer_with_weights-21
#layer-34
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_default_save_signature
+
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
20
31
B2
C3
D4
E5
L6
M7
U8
V9
k10
l11
m12
n13
v14
w15
x16
y17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67"
trackable_list_wrapper
�
20
31
B2
C3
L4
M5
U6
V7
k8
l9
v10
w11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*_default_save_signature
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
1__inference_ResNet_6channels_layer_call_fn_336836
1__inference_ResNet_6channels_layer_call_fn_338240
1__inference_ResNet_6channels_layer_call_fn_338381
1__inference_ResNet_6channels_layer_call_fn_337608�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338626
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338871
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337782
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337956�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_335565input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_1_layer_call_fn_338880�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_338890�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(�2conv2d_1/kernel
:�2conv2d_1/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_1_layer_call_fn_338895�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_338900�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_1_layer_call_fn_338913
6__inference_batch_normalization_1_layer_call_fn_338926�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338944
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338962�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_1/gamma
):'�2batch_normalization_1/beta
2:0� (2!batch_normalization_1/moving_mean
6:4� (2%batch_normalization_1/moving_variance
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_2_layer_call_fn_338971�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_338981�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_2/kernel
:�2conv2d_2/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2d_layer_call_fn_338990�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2d_layer_call_and_return_conditional_losses_339000�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
(:&�2conv2d/kernel
:�2conv2d/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_2_layer_call_fn_339005�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_339010�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_leaky_re_lu_layer_call_fn_339015�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_339020�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
k0
l1
m2
n3"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_2_layer_call_fn_339033
6__inference_batch_normalization_2_layer_call_fn_339046�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339064
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339082�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
<
v0
w1
x2
y3"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_339095
4__inference_batch_normalization_layer_call_fn_339108�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339126
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339144�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(:&�2batch_normalization/gamma
':%�2batch_normalization/beta
0:.� (2batch_normalization/moving_mean
4:2� (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_add_layer_call_fn_339150�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_add_layer_call_and_return_conditional_losses_339156�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_3_layer_call_fn_339169
6__inference_batch_normalization_3_layer_call_fn_339182�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339200
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339218�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_3/gamma
):'�2batch_normalization_3/beta
2:0� (2!batch_normalization_3/moving_mean
6:4� (2%batch_normalization_3/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_4_layer_call_fn_339227�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_339237�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_4/kernel
:�2conv2d_4/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_4_layer_call_fn_339242�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_339247�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_5_layer_call_fn_339260
6__inference_batch_normalization_5_layer_call_fn_339273�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339291
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339309�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_5/gamma
):'�2batch_normalization_5/beta
2:0� (2!batch_normalization_5/moving_mean
6:4� (2%batch_normalization_5/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_5_layer_call_fn_339318�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_339328�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_5/kernel
:�2conv2d_5/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_3_layer_call_fn_339337�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_339347�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_3/kernel
:�2conv2d_3/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_5_layer_call_fn_339352�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_339357�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_3_layer_call_fn_339362�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_339367�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_6_layer_call_fn_339380
6__inference_batch_normalization_6_layer_call_fn_339393�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339411
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339429�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_6/gamma
):'�2batch_normalization_6/beta
2:0� (2!batch_normalization_6/moving_mean
6:4� (2%batch_normalization_6/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_4_layer_call_fn_339442
6__inference_batch_normalization_4_layer_call_fn_339455�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339473
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339491�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_4/gamma
):'�2batch_normalization_4/beta
2:0� (2!batch_normalization_4/moving_mean
6:4� (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_1_layer_call_fn_339497�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_1_layer_call_and_return_conditional_losses_339503�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_7_layer_call_fn_339516
6__inference_batch_normalization_7_layer_call_fn_339529�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339547
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339565�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_7/gamma
):'�2batch_normalization_7/beta
2:0� (2!batch_normalization_7/moving_mean
6:4� (2%batch_normalization_7/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_7_layer_call_fn_339574�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_339584�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_7/kernel
:�2conv2d_7/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_7_layer_call_fn_339589�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_339594�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_9_layer_call_fn_339607
6__inference_batch_normalization_9_layer_call_fn_339620�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339638
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339656�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_9/gamma
):'�2batch_normalization_9/beta
2:0� (2!batch_normalization_9/moving_mean
6:4� (2%batch_normalization_9/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_8_layer_call_fn_339665�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_339675�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_8/kernel
:�2conv2d_8/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_6_layer_call_fn_339684�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_339694�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2conv2d_6/kernel
:�2conv2d_6/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_8_layer_call_fn_339699�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_339704�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_leaky_re_lu_6_layer_call_fn_339709�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_339714�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_10_layer_call_fn_339727
7__inference_batch_normalization_10_layer_call_fn_339740�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339758
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339776�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
+:)�2batch_normalization_10/gamma
*:(�2batch_normalization_10/beta
3:1� (2"batch_normalization_10/moving_mean
7:5� (2&batch_normalization_10/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_8_layer_call_fn_339789
6__inference_batch_normalization_8_layer_call_fn_339802�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339820
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339838�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_8/gamma
):'�2batch_normalization_8/beta
2:0� (2!batch_normalization_8/moving_mean
6:4� (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_2_layer_call_fn_339844�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_2_layer_call_and_return_conditional_losses_339850�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_11_layer_call_fn_339863
7__inference_batch_normalization_11_layer_call_fn_339876�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339894
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339912�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
+:)�2batch_normalization_11/gamma
*:(�2batch_normalization_11/beta
3:1� (2"batch_normalization_11/moving_mean
7:5� (2&batch_normalization_11/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_9_layer_call_fn_339921�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_339932�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(�2conv2d_9/kernel
:2conv2d_9/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
D0
E1
m2
n3
x4
y5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_ResNet_6channels_layer_call_fn_336836input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_ResNet_6channels_layer_call_fn_338240inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_ResNet_6channels_layer_call_fn_338381inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_ResNet_6channels_layer_call_fn_337608input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338626inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338871inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337782input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337956input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_338099input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_1_layer_call_fn_338880inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_338890inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_1_layer_call_fn_338895inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_338900inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_1_layer_call_fn_338913inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_1_layer_call_fn_338926inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338944inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338962inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_2_layer_call_fn_338971inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_338981inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2d_layer_call_fn_338990inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2d_layer_call_and_return_conditional_losses_339000inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_2_layer_call_fn_339005inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_339010inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_leaky_re_lu_layer_call_fn_339015inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_339020inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_2_layer_call_fn_339033inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_2_layer_call_fn_339046inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339064inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339082inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_layer_call_fn_339095inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_layer_call_fn_339108inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339126inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339144inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_add_layer_call_fn_339150inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_add_layer_call_and_return_conditional_losses_339156inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_3_layer_call_fn_339169inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_3_layer_call_fn_339182inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339200inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339218inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_4_layer_call_fn_339227inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_339237inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_4_layer_call_fn_339242inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_339247inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_5_layer_call_fn_339260inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_5_layer_call_fn_339273inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339291inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339309inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_5_layer_call_fn_339318inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_339328inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_3_layer_call_fn_339337inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_339347inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_5_layer_call_fn_339352inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_339357inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_3_layer_call_fn_339362inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_339367inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_6_layer_call_fn_339380inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_6_layer_call_fn_339393inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339411inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339429inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_4_layer_call_fn_339442inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_4_layer_call_fn_339455inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339473inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339491inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_1_layer_call_fn_339497inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_1_layer_call_and_return_conditional_losses_339503inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_7_layer_call_fn_339516inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_7_layer_call_fn_339529inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339547inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339565inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_7_layer_call_fn_339574inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_7_layer_call_and_return_conditional_losses_339584inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_7_layer_call_fn_339589inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_339594inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_9_layer_call_fn_339607inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_9_layer_call_fn_339620inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339638inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339656inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_8_layer_call_fn_339665inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_8_layer_call_and_return_conditional_losses_339675inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_6_layer_call_fn_339684inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_6_layer_call_and_return_conditional_losses_339694inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_8_layer_call_fn_339699inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_339704inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_leaky_re_lu_6_layer_call_fn_339709inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_339714inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_10_layer_call_fn_339727inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_10_layer_call_fn_339740inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339758inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339776inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_8_layer_call_fn_339789inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_8_layer_call_fn_339802inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339820inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339838inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_2_layer_call_fn_339844inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_2_layer_call_and_return_conditional_losses_339850inputs_0inputs_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_11_layer_call_fn_339863inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_11_layer_call_fn_339876inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339894inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339912inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_9_layer_call_fn_339921inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_339932inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337782�v23BCDEUVLMklmnvwxy��������������������������������������������������R�O
H�E
;�8
input_2+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_337956�v23BCDEUVLMklmnvwxy��������������������������������������������������R�O
H�E
;�8
input_2+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338626�v23BCDEUVLMklmnvwxy��������������������������������������������������Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
L__inference_ResNet_6channels_layer_call_and_return_conditional_losses_338871�v23BCDEUVLMklmnvwxy��������������������������������������������������Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
1__inference_ResNet_6channels_layer_call_fn_336836�v23BCDEUVLMklmnvwxy��������������������������������������������������R�O
H�E
;�8
input_2+���������������������������
p 

 
� ";�8
unknown+����������������������������
1__inference_ResNet_6channels_layer_call_fn_337608�v23BCDEUVLMklmnvwxy��������������������������������������������������R�O
H�E
;�8
input_2+���������������������������
p

 
� ";�8
unknown+����������������������������
1__inference_ResNet_6channels_layer_call_fn_338240�v23BCDEUVLMklmnvwxy��������������������������������������������������Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
1__inference_ResNet_6channels_layer_call_fn_338381�v23BCDEUVLMklmnvwxy��������������������������������������������������Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
!__inference__wrapped_model_335565�v23BCDEUVLMklmnvwxy��������������������������������������������������J�G
@�=
;�8
input_2+���������������������������
� "M�J
H
conv2d_9<�9
conv2d_9+����������������������������
A__inference_add_1_layer_call_and_return_conditional_losses_339503����
���
��~
=�:
inputs_0,����������������������������
=�:
inputs_1,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
&__inference_add_1_layer_call_fn_339497����
���
��~
=�:
inputs_0,����������������������������
=�:
inputs_1,����������������������������
� "<�9
unknown,�����������������������������
A__inference_add_2_layer_call_and_return_conditional_losses_339850����
���
��~
=�:
inputs_0,����������������������������
=�:
inputs_1,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
&__inference_add_2_layer_call_fn_339844����
���
��~
=�:
inputs_0,����������������������������
=�:
inputs_1,����������������������������
� "<�9
unknown,�����������������������������
?__inference_add_layer_call_and_return_conditional_losses_339156����
���
��~
=�:
inputs_0,����������������������������
=�:
inputs_1,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
$__inference_add_layer_call_fn_339150����
���
��~
=�:
inputs_0,����������������������������
=�:
inputs_1,����������������������������
� "<�9
unknown,�����������������������������
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339758�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_339776�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
7__inference_batch_normalization_10_layer_call_fn_339727�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
7__inference_batch_normalization_10_layer_call_fn_339740�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339894�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_339912�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
7__inference_batch_normalization_11_layer_call_fn_339863�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
7__inference_batch_normalization_11_layer_call_fn_339876�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338944�BCDEN�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_338962�BCDEN�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_1_layer_call_fn_338913�BCDEN�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_1_layer_call_fn_338926�BCDEN�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339064�klmnN�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_339082�klmnN�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_2_layer_call_fn_339033�klmnN�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_2_layer_call_fn_339046�klmnN�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339200�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_339218�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_3_layer_call_fn_339169�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_3_layer_call_fn_339182�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339473�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_339491�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_4_layer_call_fn_339442�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_4_layer_call_fn_339455�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339291�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_339309�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_5_layer_call_fn_339260�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_5_layer_call_fn_339273�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339411�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_339429�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_6_layer_call_fn_339380�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_6_layer_call_fn_339393�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339547�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_339565�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_7_layer_call_fn_339516�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_7_layer_call_fn_339529�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339820�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_339838�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_8_layer_call_fn_339789�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_8_layer_call_fn_339802�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339638�����N�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_339656�����N�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_9_layer_call_fn_339607�����N�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_9_layer_call_fn_339620�����N�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339126�vwxyN�K
D�A
;�8
inputs,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_339144�vwxyN�K
D�A
;�8
inputs,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
4__inference_batch_normalization_layer_call_fn_339095�vwxyN�K
D�A
;�8
inputs,����������������������������
p 
� "<�9
unknown,�����������������������������
4__inference_batch_normalization_layer_call_fn_339108�vwxyN�K
D�A
;�8
inputs,����������������������������
p
� "<�9
unknown,�����������������������������
D__inference_conv2d_1_layer_call_and_return_conditional_losses_338890�23I�F
?�<
:�7
inputs+���������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_1_layer_call_fn_338880�23I�F
?�<
:�7
inputs+���������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_2_layer_call_and_return_conditional_losses_338981�LMJ�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_2_layer_call_fn_338971�LMJ�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_3_layer_call_and_return_conditional_losses_339347���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_3_layer_call_fn_339337���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_4_layer_call_and_return_conditional_losses_339237���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_4_layer_call_fn_339227���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_5_layer_call_and_return_conditional_losses_339328���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_5_layer_call_fn_339318���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_6_layer_call_and_return_conditional_losses_339694���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_6_layer_call_fn_339684���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_7_layer_call_and_return_conditional_losses_339584���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_7_layer_call_fn_339574���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_8_layer_call_and_return_conditional_losses_339675���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
)__inference_conv2d_8_layer_call_fn_339665���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
D__inference_conv2d_9_layer_call_and_return_conditional_losses_339932���J�G
@�=
;�8
inputs,����������������������������
� "F�C
<�9
tensor_0+���������������������������
� �
)__inference_conv2d_9_layer_call_fn_339921���J�G
@�=
;�8
inputs,����������������������������
� ";�8
unknown+����������������������������
B__inference_conv2d_layer_call_and_return_conditional_losses_339000�UVI�F
?�<
:�7
inputs+���������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
'__inference_conv2d_layer_call_fn_338990�UVI�F
?�<
:�7
inputs+���������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_338900�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_1_layer_call_fn_338895�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_339010�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_2_layer_call_fn_339005�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_339367�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_3_layer_call_fn_339362�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_339247�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_4_layer_call_fn_339242�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_339357�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_5_layer_call_fn_339352�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_339714�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_6_layer_call_fn_339709�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_339594�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_7_layer_call_fn_339589�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_339704�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
.__inference_leaky_re_lu_8_layer_call_fn_339699�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
G__inference_leaky_re_lu_layer_call_and_return_conditional_losses_339020�J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
,__inference_leaky_re_lu_layer_call_fn_339015�J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
$__inference_signature_wrapper_338099�v23BCDEUVLMklmnvwxy��������������������������������������������������U�R
� 
K�H
F
input_2;�8
input_2+���������������������������"M�J
H
conv2d_9<�9
conv2d_9+���������������������������