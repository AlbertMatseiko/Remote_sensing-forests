Лм8
б#Ї"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
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
Ы
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
,
Cos
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
√
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
╤
ImageProjectiveTransformV3
images"dtype

transforms
output_shape

fill_value
transformed_images"dtype"
dtypetype:

2	"
interpolationstring"
	fill_modestring
CONSTANT
,
Log
x"T
y"T"
Ttype:

2
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
2
Round
x"T
y"T"
Ttype:
2
	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
,
Sin
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.12v2.11.0-94-ga3e2c692c188╗ё0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
А
Adam/v/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/v/conv2d_6/bias
y
(Adam/v/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/bias*
_output_shapes
:
*
dtype0
А
Adam/m/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/m/conv2d_6/bias
y
(Adam/m/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/bias*
_output_shapes
:
*
dtype0
Р
Adam/v/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
*'
shared_nameAdam/v/conv2d_6/kernel
Й
*Adam/v/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/kernel*&
_output_shapes
: 
*
dtype0
Р
Adam/m/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
*'
shared_nameAdam/m/conv2d_6/kernel
Й
*Adam/m/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/kernel*&
_output_shapes
: 
*
dtype0
Ъ
!Adam/v/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_7/beta
У
5Adam/v/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_7/beta*
_output_shapes
: *
dtype0
Ъ
!Adam/m/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_7/beta
У
5Adam/m/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_7/beta*
_output_shapes
: *
dtype0
Ь
"Adam/v/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_7/gamma
Х
6Adam/v/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_7/gamma*
_output_shapes
: *
dtype0
Ь
"Adam/m/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_7/gamma
Х
6Adam/m/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_7/gamma*
_output_shapes
: *
dtype0
Ъ
!Adam/v/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_4/beta
У
5Adam/v/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_4/beta*
_output_shapes
: *
dtype0
Ъ
!Adam/m/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_4/beta
У
5Adam/m/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_4/beta*
_output_shapes
: *
dtype0
Ь
"Adam/v/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_4/gamma
Х
6Adam/v/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_4/gamma*
_output_shapes
: *
dtype0
Ь
"Adam/m/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_4/gamma
Х
6Adam/m/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_4/gamma*
_output_shapes
: *
dtype0
Ъ
!Adam/v/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_6/beta
У
5Adam/v/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_6/beta*
_output_shapes
: *
dtype0
Ъ
!Adam/m/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_6/beta
У
5Adam/m/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_6/beta*
_output_shapes
: *
dtype0
Ь
"Adam/v/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_6/gamma
Х
6Adam/v/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_6/gamma*
_output_shapes
: *
dtype0
Ь
"Adam/m/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_6/gamma
Х
6Adam/m/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_6/gamma*
_output_shapes
: *
dtype0
А
Adam/v/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_3/bias
y
(Adam/v/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/bias*
_output_shapes
: *
dtype0
А
Adam/m/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_3/bias
y
(Adam/m/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/bias*
_output_shapes
: *
dtype0
Р
Adam/v/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv2d_3/kernel
Й
*Adam/v/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/kernel*&
_output_shapes
: *
dtype0
Р
Adam/m/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv2d_3/kernel
Й
*Adam/m/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/kernel*&
_output_shapes
: *
dtype0
А
Adam/v/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_5/bias
y
(Adam/v/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/bias*
_output_shapes
: *
dtype0
А
Adam/m/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_5/bias
y
(Adam/m/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/bias*
_output_shapes
: *
dtype0
Р
Adam/v/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/v/conv2d_5/kernel
Й
*Adam/v/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/kernel*&
_output_shapes
:  *
dtype0
Р
Adam/m/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/m/conv2d_5/kernel
Й
*Adam/m/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/kernel*&
_output_shapes
:  *
dtype0
Ъ
!Adam/v/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_5/beta
У
5Adam/v/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_5/beta*
_output_shapes
: *
dtype0
Ъ
!Adam/m/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_5/beta
У
5Adam/m/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_5/beta*
_output_shapes
: *
dtype0
Ь
"Adam/v/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_5/gamma
Х
6Adam/v/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_5/gamma*
_output_shapes
: *
dtype0
Ь
"Adam/m/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_5/gamma
Х
6Adam/m/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_5/gamma*
_output_shapes
: *
dtype0
А
Adam/v/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_4/bias
y
(Adam/v/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/bias*
_output_shapes
: *
dtype0
А
Adam/m/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_4/bias
y
(Adam/m/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/bias*
_output_shapes
: *
dtype0
Р
Adam/v/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv2d_4/kernel
Й
*Adam/v/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/kernel*&
_output_shapes
: *
dtype0
Р
Adam/m/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv2d_4/kernel
Й
*Adam/m/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/kernel*&
_output_shapes
: *
dtype0
Ъ
!Adam/v/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_3/beta
У
5Adam/v/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_3/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_3/beta
У
5Adam/m/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_3/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_3/gamma
Х
6Adam/v/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_3/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_3/gamma
Х
6Adam/m/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_3/gamma*
_output_shapes
:*
dtype0
Ц
Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/batch_normalization/beta
П
3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
:*
dtype0
Ц
Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/batch_normalization/beta
П
3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
:*
dtype0
Ш
 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/batch_normalization/gamma
С
4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
:*
dtype0
Ш
 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/batch_normalization/gamma
С
4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
:*
dtype0
Ъ
!Adam/v/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_2/beta
У
5Adam/v/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_2/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_2/beta
У
5Adam/m/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_2/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_2/gamma
Х
6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_2/gamma
Х
6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
|
Adam/v/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/conv2d/bias
u
&Adam/v/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/bias*
_output_shapes
:*
dtype0
|
Adam/m/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/conv2d/bias
u
&Adam/m/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/bias*
_output_shapes
:*
dtype0
М
Adam/v/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d/kernel
Е
(Adam/v/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d/kernel*&
_output_shapes
:*
dtype0
М
Adam/m/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d/kernel
Е
(Adam/m/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d/kernel*&
_output_shapes
:*
dtype0
А
Adam/v/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_2/bias
y
(Adam/v/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_2/bias
y
(Adam/m/conv2d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_2/kernel
Й
*Adam/v/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2/kernel*&
_output_shapes
:*
dtype0
Р
Adam/m/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_2/kernel
Й
*Adam/m/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2/kernel*&
_output_shapes
:*
dtype0
Ъ
!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_1/beta
У
5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_1/beta
У
5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_1/gamma
Х
6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_1/gamma
Х
6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
А
Adam/v/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_1/bias
y
(Adam/v/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/bias*
_output_shapes
:*
dtype0
А
Adam/m/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_1/bias
y
(Adam/m/conv2d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/bias*
_output_shapes
:*
dtype0
Р
Adam/v/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/conv2d_1/kernel
Й
*Adam/v/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_1/kernel*&
_output_shapes
:*
dtype0
Р
Adam/m/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/conv2d_1/kernel
Й
*Adam/m/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_1/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:
*
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 
* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: 
*
dtype0
в
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
: *
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
: *
dtype0
в
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0
в
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:  *
dtype0
в
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
: *
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
: *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
В
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
о
serving_default_input_1Placeholder*A
_output_shapes/
-:+                           *
dtype0*6
shape-:+                           
Ъ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d/kernelconv2d/biasconv2d_2/kernelconv2d_2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_5/kernelconv2d_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_6/kernelconv2d_6/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_1389914

NoOpNoOp
ЭС
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╫Р
value╠РB╚Р B└Р
ю
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%loss
&
signatures*
* 
¤
'layer-0
(layer_with_weights-0
(layer-1
)layer_with_weights-1
)layer-2
*layer-3
+layer_with_weights-2
+layer-4
,layer_with_weights-3
,layer-5
-layer_with_weights-4
-layer-6
.layer_with_weights-5
.layer-7
/layer-8
0layer-9
1layer-10
2layer_with_weights-6
2layer-11
3layer-12
4layer_with_weights-7
4layer-13
5layer_with_weights-8
5layer-14
6layer-15
7layer_with_weights-9
7layer-16
8layer_with_weights-10
8layer-17
9layer_with_weights-11
9layer-18
:layer_with_weights-12
:layer-19
;layer-20
<layer-21
=layer-22
>layer_with_weights-13
>layer-23
?layer-24
@layer_with_weights-14
@layer-25
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
О
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
О
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 

S	keras_api* 
О
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 

Z	keras_api* 

[	keras_api* 

\	keras_api* 

]	keras_api* 

^	keras_api* 

_	keras_api* 

`	keras_api* 

a	keras_api* 

b	keras_api* 

c	keras_api* 

d	keras_api* 

e	keras_api* 

f	keras_api* 

g	keras_api* 

h	keras_api* 

i	keras_api* 

j	keras_api* 

k	keras_api* 

l	keras_api* 

m	keras_api* 

n	keras_api* 
О
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses* 
Н
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
П26
Р27
С28
Т29
У30
Ф31
Х32
Ц33
Ч34
Ш35
Щ36
Ъ37
Ы38
Ь39
Э40
Ю41
Я42
а43
б44
в45*
 
u0
v1
w2
x3
{4
|5
}6
~7
8
А9
Г10
Д11
З12
И13
Л14
М15
Н16
О17
С18
Т19
У20
Ф21
Х22
Ц23
Щ24
Ъ25
Э26
Ю27
б28
в29*
* 
╡
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
:
иtrace_0
йtrace_1
кtrace_2
лtrace_3* 
:
мtrace_0
нtrace_1
оtrace_2
пtrace_3* 
* 
И
░
_variables
▒_iterations
▓_learning_rate
│_index_dict
┤
_momentums
╡_velocities
╢_update_step_xla*
* 

╖serving_default* 
* 
╧
╕	variables
╣trainable_variables
║regularization_losses
╗	keras_api
╝__call__
+╜&call_and_return_all_conditional_losses

ukernel
vbias
!╛_jit_compiled_convolution_op*
▄
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
├__call__
+─&call_and_return_all_conditional_losses
	┼axis
	wgamma
xbeta
ymoving_mean
zmoving_variance*
Ф
╞	variables
╟trainable_variables
╚regularization_losses
╔	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses* 
╧
╠	variables
═trainable_variables
╬regularization_losses
╧	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses

{kernel
|bias
!╥_jit_compiled_convolution_op*
╧
╙	variables
╘trainable_variables
╒regularization_losses
╓	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses

}kernel
~bias
!┘_jit_compiled_convolution_op*
▀
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
▐__call__
+▀&call_and_return_all_conditional_losses
	рaxis
	gamma
	Аbeta
Бmoving_mean
Вmoving_variance*
р
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
	чaxis

Гgamma
	Дbeta
Еmoving_mean
Жmoving_variance*
Ф
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses* 
Ф
ю	variables
яtrainable_variables
Ёregularization_losses
ё	keras_api
Є__call__
+є&call_and_return_all_conditional_losses* 
Ф
Ї	variables
їtrainable_variables
Ўregularization_losses
ў	keras_api
°__call__
+∙&call_and_return_all_conditional_losses* 
р
·	variables
√trainable_variables
№regularization_losses
¤	keras_api
■__call__
+ &call_and_return_all_conditional_losses
	Аaxis

Зgamma
	Иbeta
Йmoving_mean
Кmoving_variance*
Ф
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses* 
╤
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
Лkernel
	Мbias
!Н_jit_compiled_convolution_op*
р
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses
	Фaxis

Нgamma
	Оbeta
Пmoving_mean
Рmoving_variance*
Ф
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses* 
╤
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
Сkernel
	Тbias
!б_jit_compiled_convolution_op*
╤
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
Уkernel
	Фbias
!и_jit_compiled_convolution_op*
р
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses
	пaxis

Хgamma
	Цbeta
Чmoving_mean
Шmoving_variance*
р
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
┤__call__
+╡&call_and_return_all_conditional_losses
	╢axis

Щgamma
	Ъbeta
Ыmoving_mean
Ьmoving_variance*
Ф
╖	variables
╕trainable_variables
╣regularization_losses
║	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses* 
Ф
╜	variables
╛trainable_variables
┐regularization_losses
└	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses* 
Ф
├	variables
─trainable_variables
┼regularization_losses
╞	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses* 
р
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses
	╧axis

Эgamma
	Юbeta
Яmoving_mean
аmoving_variance*
Ф
╨	variables
╤trainable_variables
╥regularization_losses
╙	keras_api
╘__call__
+╒&call_and_return_all_conditional_losses* 
╤
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses
бkernel
	вbias
!▄_jit_compiled_convolution_op*
Н
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
П26
Р27
С28
Т29
У30
Ф31
Х32
Ц33
Ч34
Ш35
Щ36
Ъ37
Ы38
Ь39
Э40
Ю41
Я42
а43
б44
в45*
 
u0
v1
w2
x3
{4
|5
}6
~7
8
А9
Г10
Д11
З12
И13
Л14
М15
Н16
О17
С18
Т19
У20
Ф21
Х22
Ц23
Щ24
Ъ25
Э26
Ю27
б28
в29*
* 
Ш
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
:
тtrace_0
уtrace_1
фtrace_2
хtrace_3* 
:
цtrace_0
чtrace_1
шtrace_2
щtrace_3* 
* 
* 
* 
Ц
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

яtrace_0* 

Ёtrace_0* 
* 
* 
* 
Ц
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

Ўtrace_0* 

ўtrace_0* 
* 
* 
* 
* 
Ц
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

¤trace_0* 

■trace_0* 
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
Ц
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 

Дtrace_0* 

Еtrace_0* 
OI
VARIABLE_VALUEconv2d_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_1/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_1/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!batch_normalization_1/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%batch_normalization_1/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_2/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_2/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEbatch_normalization/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#batch_normalization/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_3/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_3/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_5/gamma'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_5/beta'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_5/moving_mean'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_5/moving_variance'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_5/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_5/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_3/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_3/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_6/gamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_6/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_6/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_6/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_4/gamma'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_4/beta'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_4/moving_mean'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_7/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_7/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_7/moving_mean'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_7/moving_variance'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_6/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_6/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
И
y0
z1
Б2
В3
Е4
Ж5
Й6
К7
П8
Р9
Ч10
Ш11
Ы12
Ь13
Я14
а15*
┌
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
27*

Ж0*
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
Я
▒0
З1
И2
Й3
К4
Л5
М6
Н7
О8
П9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
Ч17
Ш18
Щ19
Ъ20
Ы21
Ь22
Э23
Ю24
Я25
а26
б27
в28
г29
д30
е31
ж32
з33
и34
й35
к36
л37
м38
н39
о40
п41
░42
▒43
▓44
│45
┤46
╡47
╢48
╖49
╕50
╣51
║52
╗53
╝54
╜55
╛56
┐57
└58
┴59
┬60*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
И
З0
Й1
Л2
Н3
П4
С5
У6
Х7
Ч8
Щ9
Ы10
Э11
Я12
б13
г14
е15
з16
й17
л18
н19
п20
▒21
│22
╡23
╖24
╣25
╗26
╜27
┐28
┴29*
И
И0
К1
М2
О3
Р4
Т5
Ф6
Ц7
Ш8
Ъ9
Ь10
Ю11
а12
в13
д14
ж15
и16
к17
м18
о19
░20
▓21
┤22
╢23
╕24
║25
╝26
╛27
└28
┬29*
║
├trace_0
─trace_1
┼trace_2
╞trace_3
╟trace_4
╚trace_5
╔trace_6
╩trace_7
╦trace_8
╠trace_9
═trace_10
╬trace_11
╧trace_12
╨trace_13
╤trace_14
╥trace_15
╙trace_16
╘trace_17
╒trace_18
╓trace_19
╫trace_20
╪trace_21
┘trace_22
┌trace_23
█trace_24
▄trace_25
▌trace_26
▐trace_27
▀trace_28
рtrace_29* 
* 

u0
v1*

u0
v1*
* 
Ю
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
╕	variables
╣trainable_variables
║regularization_losses
╝__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses*

цtrace_0* 

чtrace_0* 
* 
 
w0
x1
y2
z3*

w0
x1*
* 
Ю
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
┐	variables
└trainable_variables
┴regularization_losses
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses*

эtrace_0
юtrace_1* 

яtrace_0
Ёtrace_1* 
* 
* 
* 
* 
Ь
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
╞	variables
╟trainable_variables
╚regularization_losses
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses* 

Ўtrace_0* 

ўtrace_0* 

{0
|1*

{0
|1*
* 
Ю
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses*

¤trace_0* 

■trace_0* 
* 

}0
~1*

}0
~1*
* 
Ю
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
╙	variables
╘trainable_variables
╒regularization_losses
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
* 
#
0
А1
Б2
В3*

0
А1*
* 
Ю
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
┌	variables
█trainable_variables
▄regularization_losses
▐__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses*

Лtrace_0
Мtrace_1* 

Нtrace_0
Оtrace_1* 
* 
$
Г0
Д1
Е2
Ж3*

Г0
Д1*
* 
Ю
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses*

Фtrace_0
Хtrace_1* 

Цtrace_0
Чtrace_1* 
* 
* 
* 
* 
Ь
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses* 

Эtrace_0* 

Юtrace_0* 
* 
* 
* 
Ь
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
ю	variables
яtrainable_variables
Ёregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses* 

дtrace_0* 

еtrace_0* 
* 
* 
* 
Ь
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
Ї	variables
їtrainable_variables
Ўregularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses* 

лtrace_0* 

мtrace_0* 
$
З0
И1
Й2
К3*

З0
И1*
* 
Ю
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
·	variables
√trainable_variables
№regularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*

▓trace_0
│trace_1* 

┤trace_0
╡trace_1* 
* 
* 
* 
* 
Ь
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses* 

╗trace_0* 

╝trace_0* 

Л0
М1*

Л0
М1*
* 
Ю
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses*

┬trace_0* 

├trace_0* 
* 
$
Н0
О1
П2
Р3*

Н0
О1*
* 
Ю
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses*

╔trace_0
╩trace_1* 

╦trace_0
╠trace_1* 
* 
* 
* 
* 
Ь
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses* 

╥trace_0* 

╙trace_0* 

С0
Т1*

С0
Т1*
* 
Ю
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses*

┘trace_0* 

┌trace_0* 
* 

У0
Ф1*

У0
Ф1*
* 
Ю
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 
* 
$
Х0
Ц1
Ч2
Ш3*

Х0
Ц1*
* 
Ю
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses*

чtrace_0
шtrace_1* 

щtrace_0
ъtrace_1* 
* 
$
Щ0
Ъ1
Ы2
Ь3*

Щ0
Ъ1*
* 
Ю
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
░	variables
▒trainable_variables
▓regularization_losses
┤__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses*

Ёtrace_0
ёtrace_1* 

Єtrace_0
єtrace_1* 
* 
* 
* 
* 
Ь
Їnon_trainable_variables
їlayers
Ўmetrics
 ўlayer_regularization_losses
°layer_metrics
╖	variables
╕trainable_variables
╣regularization_losses
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses* 

∙trace_0* 

·trace_0* 
* 
* 
* 
Ь
√non_trainable_variables
№layers
¤metrics
 ■layer_regularization_losses
 layer_metrics
╜	variables
╛trainable_variables
┐regularization_losses
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses* 

Аtrace_0* 

Бtrace_0* 
* 
* 
* 
Ь
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
├	variables
─trainable_variables
┼regularization_losses
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses* 

Зtrace_0* 

Иtrace_0* 
$
Э0
Ю1
Я2
а3*

Э0
Ю1*
* 
Ю
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses*

Оtrace_0
Пtrace_1* 

Рtrace_0
Сtrace_1* 
* 
* 
* 
* 
Ь
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
╨	variables
╤trainable_variables
╥regularization_losses
╘__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses* 

Чtrace_0* 

Шtrace_0* 

б0
в1*

б0
в1*
* 
Ю
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses*

Юtrace_0* 

Яtrace_0* 
* 
И
y0
z1
Б2
В3
Е4
Ж5
Й6
К7
П8
Р9
Ч10
Ш11
Ы12
Ь13
Я14
а15*
╩
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25*
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
* 
* 
* 
<
а	variables
б	keras_api

вtotal

гcount*
a[
VARIABLE_VALUEAdam/m/conv2d_1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv2d/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv2d/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_2/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_2/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_2/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_2/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/batch_normalization/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/batch_normalization/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/batch_normalization/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/batch_normalization/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_3/gamma2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_3/gamma2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_3/beta2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_3/beta2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_4/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_4/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_4/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_4/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_5/gamma2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_5/gamma2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_5/beta2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_5/beta2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_5/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_5/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_5/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_5/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_3/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_3/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_3/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_3/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_6/gamma2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_6/gamma2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_6/beta2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_6/beta2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_4/gamma2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_4/gamma2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_4/beta2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_4/beta2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_7/gamma2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_7/gamma2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_7/beta2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_7/beta2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_6/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_6/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_6/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_6/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
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
* 

y0
z1*
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
Б0
В1*
* 
* 
* 
* 
* 
* 
* 
* 

Е0
Ж1*
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
Й0
К1*
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
П0
Р1*
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
Ч0
Ш1*
* 
* 
* 
* 
* 
* 
* 
* 

Ы0
Ь1*
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
Я0
а1*
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
в0
г1*

а	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Я,
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*Adam/m/conv2d_1/kernel/Read/ReadVariableOp*Adam/v/conv2d_1/kernel/Read/ReadVariableOp(Adam/m/conv2d_1/bias/Read/ReadVariableOp(Adam/v/conv2d_1/bias/Read/ReadVariableOp6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_1/beta/Read/ReadVariableOp5Adam/v/batch_normalization_1/beta/Read/ReadVariableOp*Adam/m/conv2d_2/kernel/Read/ReadVariableOp*Adam/v/conv2d_2/kernel/Read/ReadVariableOp(Adam/m/conv2d_2/bias/Read/ReadVariableOp(Adam/v/conv2d_2/bias/Read/ReadVariableOp(Adam/m/conv2d/kernel/Read/ReadVariableOp(Adam/v/conv2d/kernel/Read/ReadVariableOp&Adam/m/conv2d/bias/Read/ReadVariableOp&Adam/v/conv2d/bias/Read/ReadVariableOp6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_2/beta/Read/ReadVariableOp5Adam/v/batch_normalization_2/beta/Read/ReadVariableOp4Adam/m/batch_normalization/gamma/Read/ReadVariableOp4Adam/v/batch_normalization/gamma/Read/ReadVariableOp3Adam/m/batch_normalization/beta/Read/ReadVariableOp3Adam/v/batch_normalization/beta/Read/ReadVariableOp6Adam/m/batch_normalization_3/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_3/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_3/beta/Read/ReadVariableOp5Adam/v/batch_normalization_3/beta/Read/ReadVariableOp*Adam/m/conv2d_4/kernel/Read/ReadVariableOp*Adam/v/conv2d_4/kernel/Read/ReadVariableOp(Adam/m/conv2d_4/bias/Read/ReadVariableOp(Adam/v/conv2d_4/bias/Read/ReadVariableOp6Adam/m/batch_normalization_5/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_5/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_5/beta/Read/ReadVariableOp5Adam/v/batch_normalization_5/beta/Read/ReadVariableOp*Adam/m/conv2d_5/kernel/Read/ReadVariableOp*Adam/v/conv2d_5/kernel/Read/ReadVariableOp(Adam/m/conv2d_5/bias/Read/ReadVariableOp(Adam/v/conv2d_5/bias/Read/ReadVariableOp*Adam/m/conv2d_3/kernel/Read/ReadVariableOp*Adam/v/conv2d_3/kernel/Read/ReadVariableOp(Adam/m/conv2d_3/bias/Read/ReadVariableOp(Adam/v/conv2d_3/bias/Read/ReadVariableOp6Adam/m/batch_normalization_6/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_6/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_6/beta/Read/ReadVariableOp5Adam/v/batch_normalization_6/beta/Read/ReadVariableOp6Adam/m/batch_normalization_4/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_4/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_4/beta/Read/ReadVariableOp5Adam/v/batch_normalization_4/beta/Read/ReadVariableOp6Adam/m/batch_normalization_7/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_7/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_7/beta/Read/ReadVariableOp5Adam/v/batch_normalization_7/beta/Read/ReadVariableOp*Adam/m/conv2d_6/kernel/Read/ReadVariableOp*Adam/v/conv2d_6/kernel/Read/ReadVariableOp(Adam/m/conv2d_6/bias/Read/ReadVariableOp(Adam/v/conv2d_6/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*{
Tint
r2p	*
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
GPU2 *0J 8В *)
f$R"
 __inference__traced_save_1392685
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasconv2d/kernelconv2d/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_5/kernelconv2d_5/biasconv2d_3/kernelconv2d_3/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_6/kernelconv2d_6/bias	iterationlearning_rateAdam/m/conv2d_1/kernelAdam/v/conv2d_1/kernelAdam/m/conv2d_1/biasAdam/v/conv2d_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv2d_2/kernelAdam/v/conv2d_2/kernelAdam/m/conv2d_2/biasAdam/v/conv2d_2/biasAdam/m/conv2d/kernelAdam/v/conv2d/kernelAdam/m/conv2d/biasAdam/v/conv2d/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/beta Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/beta"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/bias"Adam/m/batch_normalization_5/gamma"Adam/v/batch_normalization_5/gamma!Adam/m/batch_normalization_5/beta!Adam/v/batch_normalization_5/betaAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/biasAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/bias"Adam/m/batch_normalization_6/gamma"Adam/v/batch_normalization_6/gamma!Adam/m/batch_normalization_6/beta!Adam/v/batch_normalization_6/beta"Adam/m/batch_normalization_4/gamma"Adam/v/batch_normalization_4/gamma!Adam/m/batch_normalization_4/beta!Adam/v/batch_normalization_4/beta"Adam/m/batch_normalization_7/gamma"Adam/v/batch_normalization_7/gamma!Adam/m/batch_normalization_7/beta!Adam/v/batch_normalization_7/betaAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/biastotalcount*z
Tins
q2o*
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
GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_1393025яЦ,
Е
p
D__inference_image_projective_transform_layer_1_layer_call_fn_1391579

inputs

transforms
identityц
PartitionedCallPartitionedCallinputs
transforms*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *h
fcRa
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1388765j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         ЎЎ
:         :Y U
1
_output_shapes
:         ЎЎ

 
_user_specified_nameinputs:SO
'
_output_shapes
:         
$
_user_specified_name
transforms
╝Б
№
C__inference_ResNet_layer_call_and_return_conditional_losses_1388390
input_2*
conv2d_1_1388272:
conv2d_1_1388274:+
batch_normalization_1_1388277:+
batch_normalization_1_1388279:+
batch_normalization_1_1388281:+
batch_normalization_1_1388283:(
conv2d_1388287:
conv2d_1388289:*
conv2d_2_1388292:
conv2d_2_1388294:)
batch_normalization_1388297:)
batch_normalization_1388299:)
batch_normalization_1388301:)
batch_normalization_1388303:+
batch_normalization_2_1388306:+
batch_normalization_2_1388308:+
batch_normalization_2_1388310:+
batch_normalization_2_1388312:+
batch_normalization_3_1388318:+
batch_normalization_3_1388320:+
batch_normalization_3_1388322:+
batch_normalization_3_1388324:*
conv2d_4_1388328: 
conv2d_4_1388330: +
batch_normalization_5_1388333: +
batch_normalization_5_1388335: +
batch_normalization_5_1388337: +
batch_normalization_5_1388339: *
conv2d_3_1388343: 
conv2d_3_1388345: *
conv2d_5_1388348:  
conv2d_5_1388350: +
batch_normalization_4_1388353: +
batch_normalization_4_1388355: +
batch_normalization_4_1388357: +
batch_normalization_4_1388359: +
batch_normalization_6_1388362: +
batch_normalization_6_1388364: +
batch_normalization_6_1388366: +
batch_normalization_6_1388368: +
batch_normalization_7_1388374: +
batch_normalization_7_1388376: +
batch_normalization_7_1388378: +
batch_normalization_7_1388380: *
conv2d_6_1388384: 

conv2d_6_1388386:

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallУ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1_1388272conv2d_1_1388274*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1387382л
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1388277batch_normalization_1_1388279batch_normalization_1_1388281batch_normalization_1_1388283*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386875Р
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1387402Л
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1388287conv2d_1388289*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1387414▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1388292conv2d_2_1388294*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1387430Э
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1388297batch_normalization_1388299batch_normalization_1388301batch_normalization_1388303*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387003л
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1388306batch_normalization_2_1388308batch_normalization_2_1388310batch_normalization_2_1388312*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386939Р
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1387459К
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1387466У
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_1387474Ю
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_1388318batch_normalization_3_1388320batch_normalization_3_1388322batch_normalization_3_1388324*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387067Р
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1387490▒
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1388328conv2d_4_1388330*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1387502л
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_1388333batch_normalization_5_1388335batch_normalization_5_1388337batch_normalization_5_1388339*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387131Р
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1387522▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_3_1388343conv2d_3_1388345*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1387534▒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_5_1388348conv2d_5_1388350*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1387550л
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1388353batch_normalization_4_1388355batch_normalization_4_1388357batch_normalization_4_1388359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387259л
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_1388362batch_normalization_6_1388364batch_normalization_6_1388366batch_normalization_6_1388368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387195Р
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1387579Р
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1387586Щ
add_1/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_1387594а
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_1388374batch_normalization_7_1388376batch_normalization_7_1388378batch_normalization_7_1388380*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387323Р
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_1387610▒
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_6_1388384conv2d_6_1388386*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387623Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
╖
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_2
Ы
╘

'__inference_model_layer_call_fn_1389427
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+                           
: *@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1389233Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_1
И	
Ж
@__inference_random_affine_transform_params_layer_call_fn_1391477
inp
identity

identity_1ИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinp*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1388695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           22
StatefulPartitionedCallStatefulPartitionedCall:f b
A
_output_shapes/
-:+                           

_user_specified_nameinp
║
Я
*__inference_conv2d_6_layer_call_fn_1392321

inputs!
unknown: 

	unknown_0:

identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387623Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16868
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
═
Э
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392128

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
и
╙

'__inference_model_layer_call_fn_1390012

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╪
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+                           
: *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1388815Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╙
l
B__inference_add_1_layer_call_and_return_conditional_losses_1387594

inputs
inputs_1
identityj
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+                            i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+                            :+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386875

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16753
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Г
■
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1387382

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╧
V
"__inference__update_step_xla_16848
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╧
V
"__inference__update_step_xla_16778
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╡
e
I__inference_activation_7_layer_call_and_return_conditional_losses_1392312

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_6_layer_call_fn_1392110

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387226Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ф
J
.__inference_activation_3_layer_call_fn_1391950

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1387490z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
иБ
√
C__inference_ResNet_layer_call_and_return_conditional_losses_1388077

inputs*
conv2d_1_1387959:
conv2d_1_1387961:+
batch_normalization_1_1387964:+
batch_normalization_1_1387966:+
batch_normalization_1_1387968:+
batch_normalization_1_1387970:(
conv2d_1387974:
conv2d_1387976:*
conv2d_2_1387979:
conv2d_2_1387981:)
batch_normalization_1387984:)
batch_normalization_1387986:)
batch_normalization_1387988:)
batch_normalization_1387990:+
batch_normalization_2_1387993:+
batch_normalization_2_1387995:+
batch_normalization_2_1387997:+
batch_normalization_2_1387999:+
batch_normalization_3_1388005:+
batch_normalization_3_1388007:+
batch_normalization_3_1388009:+
batch_normalization_3_1388011:*
conv2d_4_1388015: 
conv2d_4_1388017: +
batch_normalization_5_1388020: +
batch_normalization_5_1388022: +
batch_normalization_5_1388024: +
batch_normalization_5_1388026: *
conv2d_3_1388030: 
conv2d_3_1388032: *
conv2d_5_1388035:  
conv2d_5_1388037: +
batch_normalization_4_1388040: +
batch_normalization_4_1388042: +
batch_normalization_4_1388044: +
batch_normalization_4_1388046: +
batch_normalization_6_1388049: +
batch_normalization_6_1388051: +
batch_normalization_6_1388053: +
batch_normalization_6_1388055: +
batch_normalization_7_1388061: +
batch_normalization_7_1388063: +
batch_normalization_7_1388065: +
batch_normalization_7_1388067: *
conv2d_6_1388071: 

conv2d_6_1388073:

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallТ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_1387959conv2d_1_1387961*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1387382й
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1387964batch_normalization_1_1387966batch_normalization_1_1387968batch_normalization_1_1387970*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386906Р
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1387402К
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1387974conv2d_1387976*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1387414▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1387979conv2d_2_1387981*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1387430Ы
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1387984batch_normalization_1387986batch_normalization_1387988batch_normalization_1387990*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387034й
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1387993batch_normalization_2_1387995batch_normalization_2_1387997batch_normalization_2_1387999*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386970Р
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1387459К
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1387466У
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_1387474Ь
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_1388005batch_normalization_3_1388007batch_normalization_3_1388009batch_normalization_3_1388011*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387098Р
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1387490▒
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1388015conv2d_4_1388017*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1387502й
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_1388020batch_normalization_5_1388022batch_normalization_5_1388024batch_normalization_5_1388026*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387162Р
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1387522▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_3_1388030conv2d_3_1388032*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1387534▒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_5_1388035conv2d_5_1388037*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1387550й
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1388040batch_normalization_4_1388042batch_normalization_4_1388044batch_normalization_4_1388046*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387290й
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_1388049batch_normalization_6_1388051batch_normalization_6_1388053batch_normalization_6_1388055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387226Р
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1387579Р
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1387586Щ
add_1/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_1387594Ю
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_1388061batch_normalization_7_1388063batch_normalization_7_1388065batch_normalization_7_1388067*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387354Р
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_1387610▒
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_6_1388071conv2d_6_1388073*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387623Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
╖
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
НE
З
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1391559
inp
identity

identity_1И8
ShapeShapeinp*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
random_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:З
random_uniform/RandomUniformRandomUniformrandom_uniform/shape:output:0*
T0*#
_output_shapes
:         *
dtype0c
RoundRound%random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:         J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @S
mulMul	Round:y:0mul/y:output:0*
T0*#
_output_shapes
:         J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Q
subSubmul:z:0sub/y:output:0*
T0*#
_output_shapes
:         d
random_uniform_1/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *█I└Y
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *█I@Л
random_uniform_1/RandomUniformRandomUniformrandom_uniform_1/shape:output:0*
T0*#
_output_shapes
:         *
dtype0z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: М
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:         А
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:         N
CosCosrandom_uniform_1:z:0*
T0*#
_output_shapes
:         N
SinSinrandom_uniform_1:z:0*
T0*#
_output_shapes
:         A
NegNegSin:y:0*
T0*#
_output_shapes
:         L
mul_1MulNeg:y:0sub:z:0*
T0*#
_output_shapes
:         P
Sin_1Sinrandom_uniform_1:z:0*
T0*#
_output_shapes
:         P
Cos_1Cosrandom_uniform_1:z:0*
T0*#
_output_shapes
:         N
mul_2Mul	Cos_1:y:0sub:z:0*
T0*#
_output_shapes
:         _
packed/0PackCos:y:0	mul_1:z:0*
N*
T0*'
_output_shapes
:         a
packed/1Pack	Sin_1:y:0	mul_2:z:0*
N*
T0*'
_output_shapes
:         s
packedPackpacked/0:output:0packed/1:output:0*
N*
T0*+
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:         a

packed_1/0PackCos:y:0	Sin_1:y:0*
N*
T0*'
_output_shapes
:         c

packed_1/1Pack	mul_1:z:0	mul_2:z:0*
N*
T0*'
_output_shapes
:         y
packed_1Packpacked_1/0:output:0packed_1/1:output:0*
N*
T0*+
_output_shapes
:         e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_1	Transposepacked_1:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         ^
ConstConst*
_output_shapes

:*
dtype0*!
valueB"є5Cє5C`
Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   Cn
MatMulBatchMatMulV2transpose:y:0Const_1:output:0*
T0*+
_output_shapes
:         c
sub_1SubConst:output:0MatMul:output:0*
T0*+
_output_shapes
:         k
MatMul_1BatchMatMulV2transpose_1:y:0	sub_1:z:0*
T0*+
_output_shapes
:         j
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Л
strided_slice_1StridedSliceMatMul_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskT
Neg_1Negstrided_slice_1:output:0*
T0*#
_output_shapes
:         j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Л
strided_slice_2StridedSliceMatMul_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskT
Neg_2Negstrided_slice_2:output:0*
T0*#
_output_shapes
:         E
Neg_3Neg	Neg_1:y:0*
T0*#
_output_shapes
:         N
mul_3Mul	Neg_3:y:0Cos:y:0*
T0*#
_output_shapes
:         P
mul_4Mul	Neg_2:y:0	mul_1:z:0*
T0*#
_output_shapes
:         P
sub_2Sub	mul_3:z:0	mul_4:z:0*
T0*#
_output_shapes
:         E
Neg_4Neg	Neg_2:y:0*
T0*#
_output_shapes
:         P
mul_5Mul	Neg_4:y:0	mul_2:z:0*
T0*#
_output_shapes
:         P
mul_6Mul	Neg_1:y:0	Sin_1:y:0*
T0*#
_output_shapes
:         P
sub_3Sub	mul_5:z:0	mul_6:z:0*
T0*#
_output_shapes
:         f
zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         s
zeros/ReshapeReshapestrided_slice:output:0zeros/Reshape/shape:output:0*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
zerosFillzeros/Reshape:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         h
zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         w
zeros_1/ReshapeReshapestrided_slice:output:0zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    o
zeros_1Fillzeros_1/Reshape:output:0zeros_1/Const:output:0*
T0*#
_output_shapes
:         ╢
stackPackCos:y:0	mul_1:z:0	sub_2:z:0	Sin_1:y:0	mul_2:z:0	sub_3:z:0zeros:output:0zeros_1:output:0*
N*
T0*'
_output_shapes
:         *

axish
zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         w
zeros_2/ReshapeReshapestrided_slice:output:0zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    o
zeros_2Fillzeros_2/Reshape:output:0zeros_2/Const:output:0*
T0*#
_output_shapes
:         h
zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         w
zeros_3/ReshapeReshapestrided_slice:output:0zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    o
zeros_3Fillzeros_3/Reshape:output:0zeros_3/Const:output:0*
T0*#
_output_shapes
:         ║
stack_1PackCos:y:0	Sin_1:y:0	Neg_1:y:0	mul_1:z:0	mul_2:z:0	Neg_2:y:0zeros_2:output:0zeros_3:output:0*
N*
T0*'
_output_shapes
:         *

axisX
IdentityIdentitystack_1:output:0*
T0*'
_output_shapes
:         X

Identity_1Identitystack:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :f b
A
_output_shapes/
-:+                           

_user_specified_nameinp
л╬
╗J
#__inference__traced_restore_1393025
file_prefix:
 assignvariableop_conv2d_1_kernel:.
 assignvariableop_1_conv2d_1_bias:<
.assignvariableop_2_batch_normalization_1_gamma:;
-assignvariableop_3_batch_normalization_1_beta:B
4assignvariableop_4_batch_normalization_1_moving_mean:F
8assignvariableop_5_batch_normalization_1_moving_variance:<
"assignvariableop_6_conv2d_2_kernel:.
 assignvariableop_7_conv2d_2_bias::
 assignvariableop_8_conv2d_kernel:,
assignvariableop_9_conv2d_bias:=
/assignvariableop_10_batch_normalization_2_gamma:<
.assignvariableop_11_batch_normalization_2_beta:C
5assignvariableop_12_batch_normalization_2_moving_mean:G
9assignvariableop_13_batch_normalization_2_moving_variance:;
-assignvariableop_14_batch_normalization_gamma::
,assignvariableop_15_batch_normalization_beta:A
3assignvariableop_16_batch_normalization_moving_mean:E
7assignvariableop_17_batch_normalization_moving_variance:=
/assignvariableop_18_batch_normalization_3_gamma:<
.assignvariableop_19_batch_normalization_3_beta:C
5assignvariableop_20_batch_normalization_3_moving_mean:G
9assignvariableop_21_batch_normalization_3_moving_variance:=
#assignvariableop_22_conv2d_4_kernel: /
!assignvariableop_23_conv2d_4_bias: =
/assignvariableop_24_batch_normalization_5_gamma: <
.assignvariableop_25_batch_normalization_5_beta: C
5assignvariableop_26_batch_normalization_5_moving_mean: G
9assignvariableop_27_batch_normalization_5_moving_variance: =
#assignvariableop_28_conv2d_5_kernel:  /
!assignvariableop_29_conv2d_5_bias: =
#assignvariableop_30_conv2d_3_kernel: /
!assignvariableop_31_conv2d_3_bias: =
/assignvariableop_32_batch_normalization_6_gamma: <
.assignvariableop_33_batch_normalization_6_beta: C
5assignvariableop_34_batch_normalization_6_moving_mean: G
9assignvariableop_35_batch_normalization_6_moving_variance: =
/assignvariableop_36_batch_normalization_4_gamma: <
.assignvariableop_37_batch_normalization_4_beta: C
5assignvariableop_38_batch_normalization_4_moving_mean: G
9assignvariableop_39_batch_normalization_4_moving_variance: =
/assignvariableop_40_batch_normalization_7_gamma: <
.assignvariableop_41_batch_normalization_7_beta: C
5assignvariableop_42_batch_normalization_7_moving_mean: G
9assignvariableop_43_batch_normalization_7_moving_variance: =
#assignvariableop_44_conv2d_6_kernel: 
/
!assignvariableop_45_conv2d_6_bias:
'
assignvariableop_46_iteration:	 +
!assignvariableop_47_learning_rate: D
*assignvariableop_48_adam_m_conv2d_1_kernel:D
*assignvariableop_49_adam_v_conv2d_1_kernel:6
(assignvariableop_50_adam_m_conv2d_1_bias:6
(assignvariableop_51_adam_v_conv2d_1_bias:D
6assignvariableop_52_adam_m_batch_normalization_1_gamma:D
6assignvariableop_53_adam_v_batch_normalization_1_gamma:C
5assignvariableop_54_adam_m_batch_normalization_1_beta:C
5assignvariableop_55_adam_v_batch_normalization_1_beta:D
*assignvariableop_56_adam_m_conv2d_2_kernel:D
*assignvariableop_57_adam_v_conv2d_2_kernel:6
(assignvariableop_58_adam_m_conv2d_2_bias:6
(assignvariableop_59_adam_v_conv2d_2_bias:B
(assignvariableop_60_adam_m_conv2d_kernel:B
(assignvariableop_61_adam_v_conv2d_kernel:4
&assignvariableop_62_adam_m_conv2d_bias:4
&assignvariableop_63_adam_v_conv2d_bias:D
6assignvariableop_64_adam_m_batch_normalization_2_gamma:D
6assignvariableop_65_adam_v_batch_normalization_2_gamma:C
5assignvariableop_66_adam_m_batch_normalization_2_beta:C
5assignvariableop_67_adam_v_batch_normalization_2_beta:B
4assignvariableop_68_adam_m_batch_normalization_gamma:B
4assignvariableop_69_adam_v_batch_normalization_gamma:A
3assignvariableop_70_adam_m_batch_normalization_beta:A
3assignvariableop_71_adam_v_batch_normalization_beta:D
6assignvariableop_72_adam_m_batch_normalization_3_gamma:D
6assignvariableop_73_adam_v_batch_normalization_3_gamma:C
5assignvariableop_74_adam_m_batch_normalization_3_beta:C
5assignvariableop_75_adam_v_batch_normalization_3_beta:D
*assignvariableop_76_adam_m_conv2d_4_kernel: D
*assignvariableop_77_adam_v_conv2d_4_kernel: 6
(assignvariableop_78_adam_m_conv2d_4_bias: 6
(assignvariableop_79_adam_v_conv2d_4_bias: D
6assignvariableop_80_adam_m_batch_normalization_5_gamma: D
6assignvariableop_81_adam_v_batch_normalization_5_gamma: C
5assignvariableop_82_adam_m_batch_normalization_5_beta: C
5assignvariableop_83_adam_v_batch_normalization_5_beta: D
*assignvariableop_84_adam_m_conv2d_5_kernel:  D
*assignvariableop_85_adam_v_conv2d_5_kernel:  6
(assignvariableop_86_adam_m_conv2d_5_bias: 6
(assignvariableop_87_adam_v_conv2d_5_bias: D
*assignvariableop_88_adam_m_conv2d_3_kernel: D
*assignvariableop_89_adam_v_conv2d_3_kernel: 6
(assignvariableop_90_adam_m_conv2d_3_bias: 6
(assignvariableop_91_adam_v_conv2d_3_bias: D
6assignvariableop_92_adam_m_batch_normalization_6_gamma: D
6assignvariableop_93_adam_v_batch_normalization_6_gamma: C
5assignvariableop_94_adam_m_batch_normalization_6_beta: C
5assignvariableop_95_adam_v_batch_normalization_6_beta: D
6assignvariableop_96_adam_m_batch_normalization_4_gamma: D
6assignvariableop_97_adam_v_batch_normalization_4_gamma: C
5assignvariableop_98_adam_m_batch_normalization_4_beta: C
5assignvariableop_99_adam_v_batch_normalization_4_beta: E
7assignvariableop_100_adam_m_batch_normalization_7_gamma: E
7assignvariableop_101_adam_v_batch_normalization_7_gamma: D
6assignvariableop_102_adam_m_batch_normalization_7_beta: D
6assignvariableop_103_adam_v_batch_normalization_7_beta: E
+assignvariableop_104_adam_m_conv2d_6_kernel: 
E
+assignvariableop_105_adam_v_conv2d_6_kernel: 
7
)assignvariableop_106_adam_m_conv2d_6_bias:
7
)assignvariableop_107_adam_v_conv2d_6_bias:
$
assignvariableop_108_total: $
assignvariableop_109_count: 
identity_111ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_100вAssignVariableOp_101вAssignVariableOp_102вAssignVariableOp_103вAssignVariableOp_104вAssignVariableOp_105вAssignVariableOp_106вAssignVariableOp_107вAssignVariableOp_108вAssignVariableOp_109вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98вAssignVariableOp_99█)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*Б)
valueў(BЇ(oB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*є
valueщBцoB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╠
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╥
_output_shapes┐
╝:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*}
dtypess
q2o	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_1_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_1_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_1_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv2d_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_2_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_2_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_14AssignVariableOp-assignvariableop_14_batch_normalization_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_15AssignVariableOp,assignvariableop_15_batch_normalization_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_16AssignVariableOp3assignvariableop_16_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_17AssignVariableOp7assignvariableop_17_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_3_moving_meanIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_3_moving_varianceIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_4_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_4_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_5_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_5_betaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_5_moving_meanIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_5_moving_varianceIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_5_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv2d_5_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_3_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_3_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_6_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_6_betaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_6_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_6_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_4_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_4_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_4_moving_meanIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_4_moving_varianceIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:╚
AssignVariableOp_40AssignVariableOp/assignvariableop_40_batch_normalization_7_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_41AssignVariableOp.assignvariableop_41_batch_normalization_7_betaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_42AssignVariableOp5assignvariableop_42_batch_normalization_7_moving_meanIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_43AssignVariableOp9assignvariableop_43_batch_normalization_7_moving_varianceIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_44AssignVariableOp#assignvariableop_44_conv2d_6_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_45AssignVariableOp!assignvariableop_45_conv2d_6_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_46AssignVariableOpassignvariableop_46_iterationIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_47AssignVariableOp!assignvariableop_47_learning_rateIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_conv2d_1_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_conv2d_1_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_conv2d_1_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_conv2d_1_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_m_batch_normalization_1_gammaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_v_batch_normalization_1_gammaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_m_batch_normalization_1_betaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_v_batch_normalization_1_betaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_m_conv2d_2_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_v_conv2d_2_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_m_conv2d_2_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_v_conv2d_2_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_m_conv2d_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_v_conv2d_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_62AssignVariableOp&assignvariableop_62_adam_m_conv2d_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_63AssignVariableOp&assignvariableop_63_adam_v_conv2d_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_m_batch_normalization_2_gammaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_v_batch_normalization_2_gammaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_m_batch_normalization_2_betaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_67AssignVariableOp5assignvariableop_67_adam_v_batch_normalization_2_betaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_68AssignVariableOp4assignvariableop_68_adam_m_batch_normalization_gammaIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_69AssignVariableOp4assignvariableop_69_adam_v_batch_normalization_gammaIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_70AssignVariableOp3assignvariableop_70_adam_m_batch_normalization_betaIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_71AssignVariableOp3assignvariableop_71_adam_v_batch_normalization_betaIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_m_batch_normalization_3_gammaIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_v_batch_normalization_3_gammaIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_m_batch_normalization_3_betaIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_75AssignVariableOp5assignvariableop_75_adam_v_batch_normalization_3_betaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_m_conv2d_4_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_v_conv2d_4_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_m_conv2d_4_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_v_conv2d_4_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_m_batch_normalization_5_gammaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_v_batch_normalization_5_gammaIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_m_batch_normalization_5_betaIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_83AssignVariableOp5assignvariableop_83_adam_v_batch_normalization_5_betaIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_m_conv2d_5_kernelIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_v_conv2d_5_kernelIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_m_conv2d_5_biasIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_v_conv2d_5_biasIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_88AssignVariableOp*assignvariableop_88_adam_m_conv2d_3_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_v_conv2d_3_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_m_conv2d_3_biasIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_91AssignVariableOp(assignvariableop_91_adam_v_conv2d_3_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_m_batch_normalization_6_gammaIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_v_batch_normalization_6_gammaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_m_batch_normalization_6_betaIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_95AssignVariableOp5assignvariableop_95_adam_v_batch_normalization_6_betaIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_m_batch_normalization_4_gammaIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_97AssignVariableOp6assignvariableop_97_adam_v_batch_normalization_4_gammaIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_98AssignVariableOp5assignvariableop_98_adam_m_batch_normalization_4_betaIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_99AssignVariableOp5assignvariableop_99_adam_v_batch_normalization_4_betaIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_100AssignVariableOp7assignvariableop_100_adam_m_batch_normalization_7_gammaIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:╥
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_v_batch_normalization_7_gammaIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_102AssignVariableOp6assignvariableop_102_adam_m_batch_normalization_7_betaIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_103AssignVariableOp6assignvariableop_103_adam_v_batch_normalization_7_betaIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_104AssignVariableOp+assignvariableop_104_adam_m_conv2d_6_kernelIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_v_conv2d_6_kernelIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_m_conv2d_6_biasIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_107AssignVariableOp)assignvariableop_107_adam_v_conv2d_6_biasIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_108AssignVariableOpassignvariableop_108_totalIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_109AssignVariableOpassignvariableop_109_countIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ╬
Identity_110Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_111IdentityIdentity_110:output:0^NoOp_1*
T0*
_output_shapes
: ║
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_111Identity_111:output:0*є
_input_shapesс
▐: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
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
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╢
q
E__inference_add_loss_layer_call_and_return_conditional_losses_1391598

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16873
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ф
J
.__inference_activation_6_layer_call_fn_1392213

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1387579z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
яd
з
B__inference_model_layer_call_and_return_conditional_losses_1389620
input_1(
resnet_1389430:
resnet_1389432:
resnet_1389434:
resnet_1389436:
resnet_1389438:
resnet_1389440:(
resnet_1389442:
resnet_1389444:(
resnet_1389446:
resnet_1389448:
resnet_1389450:
resnet_1389452:
resnet_1389454:
resnet_1389456:
resnet_1389458:
resnet_1389460:
resnet_1389462:
resnet_1389464:
resnet_1389466:
resnet_1389468:
resnet_1389470:
resnet_1389472:(
resnet_1389474: 
resnet_1389476: 
resnet_1389478: 
resnet_1389480: 
resnet_1389482: 
resnet_1389484: (
resnet_1389486: 
resnet_1389488: (
resnet_1389490:  
resnet_1389492: 
resnet_1389494: 
resnet_1389496: 
resnet_1389498: 
resnet_1389500: 
resnet_1389502: 
resnet_1389504: 
resnet_1389506: 
resnet_1389508: 
resnet_1389510: 
resnet_1389512: 
resnet_1389514: 
resnet_1389516: (
resnet_1389518: 

resnet_1389520:

identity

identity_1ИвResNet/StatefulPartitionedCallв ResNet/StatefulPartitionedCall_1в6random_affine_transform_params/StatefulPartitionedCallг	
ResNet/StatefulPartitionedCallStatefulPartitionedCallinput_1resnet_1389430resnet_1389432resnet_1389434resnet_1389436resnet_1389438resnet_1389440resnet_1389442resnet_1389444resnet_1389446resnet_1389448resnet_1389450resnet_1389452resnet_1389454resnet_1389456resnet_1389458resnet_1389460resnet_1389462resnet_1389464resnet_1389466resnet_1389468resnet_1389470resnet_1389472resnet_1389474resnet_1389476resnet_1389478resnet_1389480resnet_1389482resnet_1389484resnet_1389486resnet_1389488resnet_1389490resnet_1389492resnet_1389494resnet_1389496resnet_1389498resnet_1389500resnet_1389502resnet_1389504resnet_1389506resnet_1389508resnet_1389510resnet_1389512resnet_1389514resnet_1389516resnet_1389518resnet_1389520*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1387630~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+
                           П
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1388695╗
0image_projective_transform_layer/PartitionedCallPartitionedCallinput_1?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1388706О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ╡
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+
                           ╟	
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall9image_projective_transform_layer/PartitionedCall:output:0resnet_1389430resnet_1389432resnet_1389434resnet_1389436resnet_1389438resnet_1389440resnet_1389442resnet_1389444resnet_1389446resnet_1389448resnet_1389450resnet_1389452resnet_1389454resnet_1389456resnet_1389458resnet_1389460resnet_1389462resnet_1389464resnet_1389466resnet_1389468resnet_1389470resnet_1389472resnet_1389474resnet_1389476resnet_1389478resnet_1389480resnet_1389482resnet_1389484resnet_1389486resnet_1389488resnet_1389490resnet_1389492resnet_1389494resnet_1389496resnet_1389498resnet_1389500resnet_1389502resnet_1389504resnet_1389506resnet_1389508resnet_1389510resnet_1389512resnet_1389514resnet_1389516resnet_1389518resnet_1389520*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1387630с
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *h
fcRa
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1388765А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ┌
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА         
┌
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"
                  
*
paddingVALID*
strides
Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ў
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:

^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АAЦ
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:

x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       й
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:

У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:

`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:

s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         й
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        н
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:

С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:

^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:

z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:

{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"    ■   С
tf.math.reduce_sum_2/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
tf.math.reduce_mean/RankConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :│
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: ╦
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_add_loss_layer_call_and_return_conditional_losses_1388810Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ├
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_1
╢
q
E__inference_add_loss_layer_call_and_return_conditional_losses_1388810

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
╡
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1391861

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391927

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ёd
з
B__inference_model_layer_call_and_return_conditional_losses_1389813
input_1(
resnet_1389623:
resnet_1389625:
resnet_1389627:
resnet_1389629:
resnet_1389631:
resnet_1389633:(
resnet_1389635:
resnet_1389637:(
resnet_1389639:
resnet_1389641:
resnet_1389643:
resnet_1389645:
resnet_1389647:
resnet_1389649:
resnet_1389651:
resnet_1389653:
resnet_1389655:
resnet_1389657:
resnet_1389659:
resnet_1389661:
resnet_1389663:
resnet_1389665:(
resnet_1389667: 
resnet_1389669: 
resnet_1389671: 
resnet_1389673: 
resnet_1389675: 
resnet_1389677: (
resnet_1389679: 
resnet_1389681: (
resnet_1389683:  
resnet_1389685: 
resnet_1389687: 
resnet_1389689: 
resnet_1389691: 
resnet_1389693: 
resnet_1389695: 
resnet_1389697: 
resnet_1389699: 
resnet_1389701: 
resnet_1389703: 
resnet_1389705: 
resnet_1389707: 
resnet_1389709: (
resnet_1389711: 

resnet_1389713:

identity

identity_1ИвResNet/StatefulPartitionedCallв ResNet/StatefulPartitionedCall_1в6random_affine_transform_params/StatefulPartitionedCallУ	
ResNet/StatefulPartitionedCallStatefulPartitionedCallinput_1resnet_1389623resnet_1389625resnet_1389627resnet_1389629resnet_1389631resnet_1389633resnet_1389635resnet_1389637resnet_1389639resnet_1389641resnet_1389643resnet_1389645resnet_1389647resnet_1389649resnet_1389651resnet_1389653resnet_1389655resnet_1389657resnet_1389659resnet_1389661resnet_1389663resnet_1389665resnet_1389667resnet_1389669resnet_1389671resnet_1389673resnet_1389675resnet_1389677resnet_1389679resnet_1389681resnet_1389683resnet_1389685resnet_1389687resnet_1389689resnet_1389691resnet_1389693resnet_1389695resnet_1389697resnet_1389699resnet_1389701resnet_1389703resnet_1389705resnet_1389707resnet_1389709resnet_1389711resnet_1389713*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1388077~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+
                           П
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1388695╗
0image_projective_transform_layer/PartitionedCallPartitionedCallinput_1?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1388706О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ╡
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+
                           ╪	
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall9image_projective_transform_layer/PartitionedCall:output:0resnet_1389623resnet_1389625resnet_1389627resnet_1389629resnet_1389631resnet_1389633resnet_1389635resnet_1389637resnet_1389639resnet_1389641resnet_1389643resnet_1389645resnet_1389647resnet_1389649resnet_1389651resnet_1389653resnet_1389655resnet_1389657resnet_1389659resnet_1389661resnet_1389663resnet_1389665resnet_1389667resnet_1389669resnet_1389671resnet_1389673resnet_1389675resnet_1389677resnet_1389679resnet_1389681resnet_1389683resnet_1389685resnet_1389687resnet_1389689resnet_1389691resnet_1389693resnet_1389695resnet_1389697resnet_1389699resnet_1389701resnet_1389703resnet_1389705resnet_1389707resnet_1389709resnet_1389711resnet_1389713^ResNet/StatefulPartitionedCall*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ
*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1388077с
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *h
fcRa
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1388765А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ┌
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА         
┌
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"
                  
*
paddingVALID*
strides
Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ў
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:

^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АAЦ
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:

x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       й
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:

У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:

`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:

s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         й
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        н
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:

С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:

^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:

z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:

{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"    ■   С
tf.math.reduce_sum_2/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
tf.math.reduce_mean/RankConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :│
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: ╦
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_add_loss_layer_call_and_return_conditional_losses_1388810Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ├
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_1
Г
■
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1387430

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_2_layer_call_fn_1391753

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386970Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392190

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ф
J
.__inference_activation_7_layer_call_fn_1392307

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_1387610z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
║
Я
*__inference_conv2d_3_layer_call_fn_1392074

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1387534Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391789

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┘
l
@__inference_add_layer_call_and_return_conditional_losses_1391883
inputs_0
inputs_1
identityl
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+                           i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+                           :+                           :k g
A
_output_shapes/
-:+                           
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+                           
"
_user_specified_name
inputs_1
═
Э
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386939

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386906

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16878
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ч
╘

(__inference_ResNet_layer_call_fn_1391132

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╞
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1388077Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16763
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
─
Л
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1388765

inputs

transforms
identityx
'ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      j
%ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
ImageProjectiveTransformV3ImageProjectiveTransformV3inputs
transforms0ImageProjectiveTransformV3/output_shape:output:0.ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         АА
*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:         АА
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         ЎЎ
:         :Y U
1
_output_shapes
:         ЎЎ

 
_user_specified_nameinputs:SO
'
_output_shapes
:         
$
_user_specified_name
transforms
З
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387354

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16833
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
к
╒

(__inference_ResNet_layer_call_fn_1387725
input_2!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╫
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1387630Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_2
л
J
"__inference__update_step_xla_16828
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ц	
╥
7__inference_batch_normalization_3_layer_call_fn_1391909

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387098Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╦
Ы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391833

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┘Л
╪-
C__inference_ResNet_layer_call_and_return_conditional_losses_1391470

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: ;
-batch_normalization_5_readvariableop_resource: =
/batch_normalization_5_readvariableop_1_resource: L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource:  6
(conv2d_5_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: 
6
(conv2d_6_biasadd_readvariableop_resource:

identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1в$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1в$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в$batch_normalization_4/AssignNewValueв&batch_normalization_4/AssignNewValue_1в5batch_normalization_4/FusedBatchNormV3/ReadVariableOpв7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_4/ReadVariableOpв&batch_normalization_4/ReadVariableOp_1в$batch_normalization_5/AssignNewValueв&batch_normalization_5/AssignNewValue_1в5batch_normalization_5/FusedBatchNormV3/ReadVariableOpв7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_5/ReadVariableOpв&batch_normalization_5/ReadVariableOp_1в$batch_normalization_6/AssignNewValueв&batch_normalization_6/AssignNewValue_1в5batch_normalization_6/FusedBatchNormV3/ReadVariableOpв7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_6/ReadVariableOpв&batch_normalization_6/ReadVariableOp_1в$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1вconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOpвconv2d_4/BiasAdd/ReadVariableOpвconv2d_4/Conv2D/ReadVariableOpвconv2d_5/BiasAdd/ReadVariableOpвconv2d_5/Conv2D/ReadVariableOpвconv2d_6/BiasAdd/ReadVariableOpвconv2d_6/Conv2D/ReadVariableOpО
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╜
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0к
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╫
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(С
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╣
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╓
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0к
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0м
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╦
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ц
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╫
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(С
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Н
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ь
add/addAddV2activation_2/Relu:activations:0activation/Relu:activations:0*
T0*A
_output_shapes/
-:+                           О
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╔
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3add/add:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(С
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           О
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╓
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Д
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            О
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╫
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(С
activation_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            О
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╓
conv2d_3/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Д
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            О
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0╓
conv2d_5/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            О
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╫
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(О
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╫
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(С
activation_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            С
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            а
	add_1/addAddV2activation_6/Relu:activations:0activation_4/Relu:activations:0*
T0*A
_output_shapes/
-:+                            О
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╦
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3add_1/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(С
activation_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0╓
conv2d_6/Conv2DConv2Dactivation_7/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0к
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
В
conv2d_6/SoftmaxSoftmaxconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           
Г
IdentityIdentityconv2d_6/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+                           
Ы
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
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
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12>
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
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Г
■
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1391617

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ў
■
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387623

inputs8
conv2d_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
p
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           
z
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+                           
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╧
V
"__inference__update_step_xla_16838
gradient"
variable:  *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:  : *
	_noinline(:P L
&
_output_shapes
:  
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╧
V
"__inference__update_step_xla_16888
gradient"
variable: 
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: 
: *
	_noinline(:P L
&
_output_shapes
: 

"
_user_specified_name
gradient:($
"
_user_specified_name
variable
═
Э
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391661

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш	
╥
7__inference_batch_normalization_6_layer_call_fn_1392097

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387195Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ыd
ж
B__inference_model_layer_call_and_return_conditional_losses_1389233

inputs(
resnet_1389043:
resnet_1389045:
resnet_1389047:
resnet_1389049:
resnet_1389051:
resnet_1389053:(
resnet_1389055:
resnet_1389057:(
resnet_1389059:
resnet_1389061:
resnet_1389063:
resnet_1389065:
resnet_1389067:
resnet_1389069:
resnet_1389071:
resnet_1389073:
resnet_1389075:
resnet_1389077:
resnet_1389079:
resnet_1389081:
resnet_1389083:
resnet_1389085:(
resnet_1389087: 
resnet_1389089: 
resnet_1389091: 
resnet_1389093: 
resnet_1389095: 
resnet_1389097: (
resnet_1389099: 
resnet_1389101: (
resnet_1389103:  
resnet_1389105: 
resnet_1389107: 
resnet_1389109: 
resnet_1389111: 
resnet_1389113: 
resnet_1389115: 
resnet_1389117: 
resnet_1389119: 
resnet_1389121: 
resnet_1389123: 
resnet_1389125: 
resnet_1389127: 
resnet_1389129: (
resnet_1389131: 

resnet_1389133:

identity

identity_1ИвResNet/StatefulPartitionedCallв ResNet/StatefulPartitionedCall_1в6random_affine_transform_params/StatefulPartitionedCallТ	
ResNet/StatefulPartitionedCallStatefulPartitionedCallinputsresnet_1389043resnet_1389045resnet_1389047resnet_1389049resnet_1389051resnet_1389053resnet_1389055resnet_1389057resnet_1389059resnet_1389061resnet_1389063resnet_1389065resnet_1389067resnet_1389069resnet_1389071resnet_1389073resnet_1389075resnet_1389077resnet_1389079resnet_1389081resnet_1389083resnet_1389085resnet_1389087resnet_1389089resnet_1389091resnet_1389093resnet_1389095resnet_1389097resnet_1389099resnet_1389101resnet_1389103resnet_1389105resnet_1389107resnet_1389109resnet_1389111resnet_1389113resnet_1389115resnet_1389117resnet_1389119resnet_1389121resnet_1389123resnet_1389125resnet_1389127resnet_1389129resnet_1389131resnet_1389133*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1388077~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+
                           О
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1388695║
0image_projective_transform_layer/PartitionedCallPartitionedCallinputs?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1388706О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ╡
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+
                           ╪	
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall9image_projective_transform_layer/PartitionedCall:output:0resnet_1389043resnet_1389045resnet_1389047resnet_1389049resnet_1389051resnet_1389053resnet_1389055resnet_1389057resnet_1389059resnet_1389061resnet_1389063resnet_1389065resnet_1389067resnet_1389069resnet_1389071resnet_1389073resnet_1389075resnet_1389077resnet_1389079resnet_1389081resnet_1389083resnet_1389085resnet_1389087resnet_1389089resnet_1389091resnet_1389093resnet_1389095resnet_1389097resnet_1389099resnet_1389101resnet_1389103resnet_1389105resnet_1389107resnet_1389109resnet_1389111resnet_1389113resnet_1389115resnet_1389117resnet_1389119resnet_1389121resnet_1389123resnet_1389125resnet_1389127resnet_1389129resnet_1389131resnet_1389133^ResNet/StatefulPartitionedCall*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ
*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1388077с
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *h
fcRa
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1388765А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ┌
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА         
┌
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"
                  
*
paddingVALID*
strides
Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ў
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:

^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АAЦ
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:

x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       й
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:

У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:

`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:

s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         й
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        н
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:

С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:

^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:

z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:

{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"    ■   С
tf.math.reduce_sum_2/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
tf.math.reduce_mean/RankConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :│
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: ╦
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_add_loss_layer_call_and_return_conditional_losses_1388810Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ├
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16893
gradient
variable:
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:
: *
	_noinline(:D @

_output_shapes
:

"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ш	
╥
7__inference_batch_normalization_3_layer_call_fn_1391896

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387067Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16853
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╡
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1391955

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╡
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1391689

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392302

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╡
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1387459

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ъd
ж
B__inference_model_layer_call_and_return_conditional_losses_1388815

inputs(
resnet_1388518:
resnet_1388520:
resnet_1388522:
resnet_1388524:
resnet_1388526:
resnet_1388528:(
resnet_1388530:
resnet_1388532:(
resnet_1388534:
resnet_1388536:
resnet_1388538:
resnet_1388540:
resnet_1388542:
resnet_1388544:
resnet_1388546:
resnet_1388548:
resnet_1388550:
resnet_1388552:
resnet_1388554:
resnet_1388556:
resnet_1388558:
resnet_1388560:(
resnet_1388562: 
resnet_1388564: 
resnet_1388566: 
resnet_1388568: 
resnet_1388570: 
resnet_1388572: (
resnet_1388574: 
resnet_1388576: (
resnet_1388578:  
resnet_1388580: 
resnet_1388582: 
resnet_1388584: 
resnet_1388586: 
resnet_1388588: 
resnet_1388590: 
resnet_1388592: 
resnet_1388594: 
resnet_1388596: 
resnet_1388598: 
resnet_1388600: 
resnet_1388602: 
resnet_1388604: (
resnet_1388606: 

resnet_1388608:

identity

identity_1ИвResNet/StatefulPartitionedCallв ResNet/StatefulPartitionedCall_1в6random_affine_transform_params/StatefulPartitionedCallв	
ResNet/StatefulPartitionedCallStatefulPartitionedCallinputsresnet_1388518resnet_1388520resnet_1388522resnet_1388524resnet_1388526resnet_1388528resnet_1388530resnet_1388532resnet_1388534resnet_1388536resnet_1388538resnet_1388540resnet_1388542resnet_1388544resnet_1388546resnet_1388548resnet_1388550resnet_1388552resnet_1388554resnet_1388556resnet_1388558resnet_1388560resnet_1388562resnet_1388564resnet_1388566resnet_1388568resnet_1388570resnet_1388572resnet_1388574resnet_1388576resnet_1388578resnet_1388580resnet_1388582resnet_1388584resnet_1388586resnet_1388588resnet_1388590resnet_1388592resnet_1388594resnet_1388596resnet_1388598resnet_1388600resnet_1388602resnet_1388604resnet_1388606resnet_1388608*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1387630~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+
                           О
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1388695║
0image_projective_transform_layer/PartitionedCallPartitionedCallinputs?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1388706О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ╡
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+
                           ╟	
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall9image_projective_transform_layer/PartitionedCall:output:0resnet_1388518resnet_1388520resnet_1388522resnet_1388524resnet_1388526resnet_1388528resnet_1388530resnet_1388532resnet_1388534resnet_1388536resnet_1388538resnet_1388540resnet_1388542resnet_1388544resnet_1388546resnet_1388548resnet_1388550resnet_1388552resnet_1388554resnet_1388556resnet_1388558resnet_1388560resnet_1388562resnet_1388564resnet_1388566resnet_1388568resnet_1388570resnet_1388572resnet_1388574resnet_1388576resnet_1388578resnet_1388580resnet_1388582resnet_1388584resnet_1388586resnet_1388588resnet_1388590resnet_1388592resnet_1388594resnet_1388596resnet_1388598resnet_1388600resnet_1388602resnet_1388604resnet_1388606resnet_1388608*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1387630с
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *h
fcRa
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1388765А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ┌
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА         
┌
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"
                  
*
paddingVALID*
strides
Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ў
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:

^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АAЦ
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:

x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       й
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:

У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:

`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:

s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         й
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        н
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:

С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:

^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:

z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:

{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"    ■   С
tf.math.reduce_sum_2/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
tf.math.reduce_mean/RankConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :│
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: ╦
add_loss/PartitionedCallPartitionedCall!tf.math.reduce_mean/Mean:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_add_loss_layer_call_and_return_conditional_losses_1388810Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ├
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Г
■
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1392084

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
║
Я
*__inference_conv2d_4_layer_call_fn_1391964

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1387502Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╡
e
I__inference_activation_6_layer_call_and_return_conditional_losses_1392218

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
б
n
B__inference_image_projective_transform_layer_layer_call_fn_1391565

inputs

transforms
identityф
PartitionedCallPartitionedCallinputs
transforms*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЎЎ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1388706j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЎЎ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:+                           :         :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:SO
'
_output_shapes
:         
$
_user_specified_name
transforms
Ш	
╥
7__inference_batch_normalization_5_layer_call_fn_1391987

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387131Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16788
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ж
╥

%__inference_signature_wrapper_1389914
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_1386853Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_1
З
┴
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391679

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387226

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Е
┐
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391851

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
║
Я
*__inference_conv2d_1_layer_call_fn_1391607

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1387382Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_5_layer_call_fn_1392000

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387162Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ф
J
.__inference_activation_2_layer_call_fn_1391856

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1387459z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Г
■
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1387534

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╧
V
"__inference__update_step_xla_16748
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Р
H
,__inference_activation_layer_call_fn_1391866

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1387466z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┐
S
'__inference_add_1_layer_call_fn_1392234
inputs_0
inputs_1
identity┘
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_1387594z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+                            :+                            :k g
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs_1
мБ
№
C__inference_ResNet_layer_call_and_return_conditional_losses_1388511
input_2*
conv2d_1_1388393:
conv2d_1_1388395:+
batch_normalization_1_1388398:+
batch_normalization_1_1388400:+
batch_normalization_1_1388402:+
batch_normalization_1_1388404:(
conv2d_1388408:
conv2d_1388410:*
conv2d_2_1388413:
conv2d_2_1388415:)
batch_normalization_1388418:)
batch_normalization_1388420:)
batch_normalization_1388422:)
batch_normalization_1388424:+
batch_normalization_2_1388427:+
batch_normalization_2_1388429:+
batch_normalization_2_1388431:+
batch_normalization_2_1388433:+
batch_normalization_3_1388439:+
batch_normalization_3_1388441:+
batch_normalization_3_1388443:+
batch_normalization_3_1388445:*
conv2d_4_1388449: 
conv2d_4_1388451: +
batch_normalization_5_1388454: +
batch_normalization_5_1388456: +
batch_normalization_5_1388458: +
batch_normalization_5_1388460: *
conv2d_3_1388464: 
conv2d_3_1388466: *
conv2d_5_1388469:  
conv2d_5_1388471: +
batch_normalization_4_1388474: +
batch_normalization_4_1388476: +
batch_normalization_4_1388478: +
batch_normalization_4_1388480: +
batch_normalization_6_1388483: +
batch_normalization_6_1388485: +
batch_normalization_6_1388487: +
batch_normalization_6_1388489: +
batch_normalization_7_1388495: +
batch_normalization_7_1388497: +
batch_normalization_7_1388499: +
batch_normalization_7_1388501: *
conv2d_6_1388505: 

conv2d_6_1388507:

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallУ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1_1388393conv2d_1_1388395*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1387382й
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1388398batch_normalization_1_1388400batch_normalization_1_1388402batch_normalization_1_1388404*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386906Р
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1387402Л
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1388408conv2d_1388410*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1387414▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1388413conv2d_2_1388415*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1387430Ы
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1388418batch_normalization_1388420batch_normalization_1388422batch_normalization_1388424*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387034й
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1388427batch_normalization_2_1388429batch_normalization_2_1388431batch_normalization_2_1388433*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386970Р
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1387459К
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1387466У
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_1387474Ь
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_1388439batch_normalization_3_1388441batch_normalization_3_1388443batch_normalization_3_1388445*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387098Р
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1387490▒
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1388449conv2d_4_1388451*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1387502й
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_1388454batch_normalization_5_1388456batch_normalization_5_1388458batch_normalization_5_1388460*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387162Р
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1387522▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_3_1388464conv2d_3_1388466*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1387534▒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_5_1388469conv2d_5_1388471*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1387550й
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1388474batch_normalization_4_1388476batch_normalization_4_1388478batch_normalization_4_1388480*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387290й
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_1388483batch_normalization_6_1388485batch_normalization_6_1388487batch_normalization_6_1388489*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387226Р
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1387579Р
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1387586Щ
add_1/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_1387594Ю
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_1388495batch_normalization_7_1388497batch_normalization_7_1388499batch_normalization_7_1388501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387354Р
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_1387610▒
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_6_1388505conv2d_6_1388507*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387623Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
╖
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_2
З
┴
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387290

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╧
V
"__inference__update_step_xla_16768
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Г
■
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1391974

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
║
Я
*__inference_conv2d_2_layer_call_fn_1391698

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1387430Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16758
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
 С
уF
"__inference__wrapped_model_1386853
input_1N
4model_resnet_conv2d_1_conv2d_readvariableop_resource:C
5model_resnet_conv2d_1_biasadd_readvariableop_resource:H
:model_resnet_batch_normalization_1_readvariableop_resource:J
<model_resnet_batch_normalization_1_readvariableop_1_resource:Y
Kmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:[
Mmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:L
2model_resnet_conv2d_conv2d_readvariableop_resource:A
3model_resnet_conv2d_biasadd_readvariableop_resource:N
4model_resnet_conv2d_2_conv2d_readvariableop_resource:C
5model_resnet_conv2d_2_biasadd_readvariableop_resource:F
8model_resnet_batch_normalization_readvariableop_resource:H
:model_resnet_batch_normalization_readvariableop_1_resource:W
Imodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:Y
Kmodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H
:model_resnet_batch_normalization_2_readvariableop_resource:J
<model_resnet_batch_normalization_2_readvariableop_1_resource:Y
Kmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Mmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:H
:model_resnet_batch_normalization_3_readvariableop_resource:J
<model_resnet_batch_normalization_3_readvariableop_1_resource:Y
Kmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:[
Mmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:N
4model_resnet_conv2d_4_conv2d_readvariableop_resource: C
5model_resnet_conv2d_4_biasadd_readvariableop_resource: H
:model_resnet_batch_normalization_5_readvariableop_resource: J
<model_resnet_batch_normalization_5_readvariableop_1_resource: Y
Kmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource: [
Mmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: N
4model_resnet_conv2d_3_conv2d_readvariableop_resource: C
5model_resnet_conv2d_3_biasadd_readvariableop_resource: N
4model_resnet_conv2d_5_conv2d_readvariableop_resource:  C
5model_resnet_conv2d_5_biasadd_readvariableop_resource: H
:model_resnet_batch_normalization_4_readvariableop_resource: J
<model_resnet_batch_normalization_4_readvariableop_1_resource: Y
Kmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: [
Mmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: H
:model_resnet_batch_normalization_6_readvariableop_resource: J
<model_resnet_batch_normalization_6_readvariableop_1_resource: Y
Kmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: [
Mmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: H
:model_resnet_batch_normalization_7_readvariableop_resource: J
<model_resnet_batch_normalization_7_readvariableop_1_resource: Y
Kmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource: [
Mmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: N
4model_resnet_conv2d_6_conv2d_readvariableop_resource: 
C
5model_resnet_conv2d_6_biasadd_readvariableop_resource:

identityИв@model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpвBmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1вBmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpвDmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1в/model/ResNet/batch_normalization/ReadVariableOpв1model/ResNet/batch_normalization/ReadVariableOp_1в1model/ResNet/batch_normalization/ReadVariableOp_2в1model/ResNet/batch_normalization/ReadVariableOp_3вBmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвDmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1вDmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpвFmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1в1model/ResNet/batch_normalization_1/ReadVariableOpв3model/ResNet/batch_normalization_1/ReadVariableOp_1в3model/ResNet/batch_normalization_1/ReadVariableOp_2в3model/ResNet/batch_normalization_1/ReadVariableOp_3вBmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1вDmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpвFmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1в1model/ResNet/batch_normalization_2/ReadVariableOpв3model/ResNet/batch_normalization_2/ReadVariableOp_1в3model/ResNet/batch_normalization_2/ReadVariableOp_2в3model/ResNet/batch_normalization_2/ReadVariableOp_3вBmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpвDmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1вDmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpвFmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1в1model/ResNet/batch_normalization_3/ReadVariableOpв3model/ResNet/batch_normalization_3/ReadVariableOp_1в3model/ResNet/batch_normalization_3/ReadVariableOp_2в3model/ResNet/batch_normalization_3/ReadVariableOp_3вBmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpвDmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1вDmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpвFmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1в1model/ResNet/batch_normalization_4/ReadVariableOpв3model/ResNet/batch_normalization_4/ReadVariableOp_1в3model/ResNet/batch_normalization_4/ReadVariableOp_2в3model/ResNet/batch_normalization_4/ReadVariableOp_3вBmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1вDmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpвFmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1в1model/ResNet/batch_normalization_5/ReadVariableOpв3model/ResNet/batch_normalization_5/ReadVariableOp_1в3model/ResNet/batch_normalization_5/ReadVariableOp_2в3model/ResNet/batch_normalization_5/ReadVariableOp_3вBmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpвDmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1вDmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpвFmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1в1model/ResNet/batch_normalization_6/ReadVariableOpв3model/ResNet/batch_normalization_6/ReadVariableOp_1в3model/ResNet/batch_normalization_6/ReadVariableOp_2в3model/ResNet/batch_normalization_6/ReadVariableOp_3вBmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1вDmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpвFmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1в1model/ResNet/batch_normalization_7/ReadVariableOpв3model/ResNet/batch_normalization_7/ReadVariableOp_1в3model/ResNet/batch_normalization_7/ReadVariableOp_2в3model/ResNet/batch_normalization_7/ReadVariableOp_3в*model/ResNet/conv2d/BiasAdd/ReadVariableOpв,model/ResNet/conv2d/BiasAdd_1/ReadVariableOpв)model/ResNet/conv2d/Conv2D/ReadVariableOpв+model/ResNet/conv2d/Conv2D_1/ReadVariableOpв,model/ResNet/conv2d_1/BiasAdd/ReadVariableOpв.model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOpв+model/ResNet/conv2d_1/Conv2D/ReadVariableOpв-model/ResNet/conv2d_1/Conv2D_1/ReadVariableOpв,model/ResNet/conv2d_2/BiasAdd/ReadVariableOpв.model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOpв+model/ResNet/conv2d_2/Conv2D/ReadVariableOpв-model/ResNet/conv2d_2/Conv2D_1/ReadVariableOpв,model/ResNet/conv2d_3/BiasAdd/ReadVariableOpв.model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOpв+model/ResNet/conv2d_3/Conv2D/ReadVariableOpв-model/ResNet/conv2d_3/Conv2D_1/ReadVariableOpв,model/ResNet/conv2d_4/BiasAdd/ReadVariableOpв.model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOpв+model/ResNet/conv2d_4/Conv2D/ReadVariableOpв-model/ResNet/conv2d_4/Conv2D_1/ReadVariableOpв,model/ResNet/conv2d_5/BiasAdd/ReadVariableOpв.model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOpв+model/ResNet/conv2d_5/Conv2D/ReadVariableOpв-model/ResNet/conv2d_5/Conv2D_1/ReadVariableOpв,model/ResNet/conv2d_6/BiasAdd/ReadVariableOpв.model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOpв+model/ResNet/conv2d_6/Conv2D/ReadVariableOpв-model/ResNet/conv2d_6/Conv2D_1/ReadVariableOpи
+model/ResNet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╪
model/ResNet/conv2d_1/Conv2DConv2Dinput_13model/ResNet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Ю
,model/ResNet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╤
model/ResNet/conv2d_1/BiasAddBiasAdd%model/ResNet/conv2d_1/Conv2D:output:04model/ResNet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           и
1model/ResNet/batch_normalization_1/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0м
3model/ResNet/batch_normalization_1/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ч
3model/ResNet/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_1/BiasAdd:output:09model/ResNet/batch_normalization_1/ReadVariableOp:value:0;model/ResNet/batch_normalization_1/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( л
model/ResNet/activation_1/ReluRelu7model/ResNet/batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           д
)model/ResNet/conv2d/Conv2D/ReadVariableOpReadVariableOp2model_resnet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╘
model/ResNet/conv2d/Conv2DConv2Dinput_11model/ResNet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Ъ
*model/ResNet/conv2d/BiasAdd/ReadVariableOpReadVariableOp3model_resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╦
model/ResNet/conv2d/BiasAddBiasAdd#model/ResNet/conv2d/Conv2D:output:02model/ResNet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           и
+model/ResNet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¤
model/ResNet/conv2d_2/Conv2DConv2D,model/ResNet/activation_1/Relu:activations:03model/ResNet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Ю
,model/ResNet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╤
model/ResNet/conv2d_2/BiasAddBiasAdd%model/ResNet/conv2d_2/Conv2D:output:04model/ResNet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           д
/model/ResNet/batch_normalization/ReadVariableOpReadVariableOp8model_resnet_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0и
1model/ResNet/batch_normalization/ReadVariableOp_1ReadVariableOp:model_resnet_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0╞
@model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╩
Bmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
1model/ResNet/batch_normalization/FusedBatchNormV3FusedBatchNormV3$model/ResNet/conv2d/BiasAdd:output:07model/ResNet/batch_normalization/ReadVariableOp:value:09model/ResNet/batch_normalization/ReadVariableOp_1:value:0Hmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( и
1model/ResNet/batch_normalization_2/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3model/ResNet/batch_normalization_2/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ч
3model/ResNet/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_2/BiasAdd:output:09model/ResNet/batch_normalization_2/ReadVariableOp:value:0;model/ResNet/batch_normalization_2/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( л
model/ResNet/activation_2/ReluRelu7model/ResNet/batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           з
model/ResNet/activation/ReluRelu5model/ResNet/batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           ├
model/ResNet/add/addAddV2,model/ResNet/activation_2/Relu:activations:0*model/ResNet/activation/Relu:activations:0*
T0*A
_output_shapes/
-:+                           и
1model/ResNet/batch_normalization_3/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0м
3model/ResNet/batch_normalization_3/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Й
3model/ResNet/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3model/ResNet/add/add:z:09model/ResNet/batch_normalization_3/ReadVariableOp:value:0;model/ResNet/batch_normalization_3/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( л
model/ResNet/activation_3/ReluRelu7model/ResNet/batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           и
+model/ResNet/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¤
model/ResNet/conv2d_4/Conv2DConv2D,model/ResNet/activation_3/Relu:activations:03model/ResNet/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Ю
,model/ResNet/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╤
model/ResNet/conv2d_4/BiasAddBiasAdd%model/ResNet/conv2d_4/Conv2D:output:04model/ResNet/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            и
1model/ResNet/batch_normalization_5/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_5/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0╩
Bmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╬
Dmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ч
3model/ResNet/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_4/BiasAdd:output:09model/ResNet/batch_normalization_5/ReadVariableOp:value:0;model/ResNet/batch_normalization_5/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( л
model/ResNet/activation_5/ReluRelu7model/ResNet/batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            и
+model/ResNet/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¤
model/ResNet/conv2d_3/Conv2DConv2D,model/ResNet/activation_3/Relu:activations:03model/ResNet/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Ю
,model/ResNet/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╤
model/ResNet/conv2d_3/BiasAddBiasAdd%model/ResNet/conv2d_3/Conv2D:output:04model/ResNet/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            и
+model/ResNet/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0¤
model/ResNet/conv2d_5/Conv2DConv2D,model/ResNet/activation_5/Relu:activations:03model/ResNet/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Ю
,model/ResNet/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╤
model/ResNet/conv2d_5/BiasAddBiasAdd%model/ResNet/conv2d_5/Conv2D:output:04model/ResNet/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            и
1model/ResNet/batch_normalization_4/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_4/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0╩
Bmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╬
Dmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ч
3model/ResNet/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_3/BiasAdd:output:09model/ResNet/batch_normalization_4/ReadVariableOp:value:0;model/ResNet/batch_normalization_4/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( и
1model/ResNet/batch_normalization_6/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_6/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0╩
Bmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╬
Dmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ч
3model/ResNet/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_5/BiasAdd:output:09model/ResNet/batch_normalization_6/ReadVariableOp:value:0;model/ResNet/batch_normalization_6/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( л
model/ResNet/activation_6/ReluRelu7model/ResNet/batch_normalization_6/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            л
model/ResNet/activation_4/ReluRelu7model/ResNet/batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            ╟
model/ResNet/add_1/addAddV2,model/ResNet/activation_6/Relu:activations:0,model/ResNet/activation_4/Relu:activations:0*
T0*A
_output_shapes/
-:+                            и
1model/ResNet/batch_normalization_7/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_7/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0╩
Bmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╬
Dmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Л
3model/ResNet/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3model/ResNet/add_1/add:z:09model/ResNet/batch_normalization_7/ReadVariableOp:value:0;model/ResNet/batch_normalization_7/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( л
model/ResNet/activation_7/ReluRelu7model/ResNet/batch_normalization_7/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            и
+model/ResNet/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0¤
model/ResNet/conv2d_6/Conv2DConv2D,model/ResNet/activation_7/Relu:activations:03model/ResNet/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
*
paddingSAME*
strides
Ю
,model/ResNet/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╤
model/ResNet/conv2d_6/BiasAddBiasAdd%model/ResNet/conv2d_6/Conv2D:output:04model/ResNet/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
Ь
model/ResNet/conv2d_6/SoftmaxSoftmax&model/ResNet/conv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           
Д
+model/tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ▐
&model/tf.compat.v1.transpose/transpose	Transpose'model/ResNet/conv2d_6/Softmax:softmax:04model/tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+
                           a
*model/random_affine_transform_params/ShapeShapeinput_1*
T0*
_output_shapes
:В
8model/random_affine_transform_params/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
:model/random_affine_transform_params/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
:model/random_affine_transform_params/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
2model/random_affine_transform_params/strided_sliceStridedSlice3model/random_affine_transform_params/Shape:output:0Amodel/random_affine_transform_params/strided_slice/stack:output:0Cmodel/random_affine_transform_params/strided_slice/stack_1:output:0Cmodel/random_affine_transform_params/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskм
9model/random_affine_transform_params/random_uniform/shapePack;model/random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:╤
Amodel/random_affine_transform_params/random_uniform/RandomUniformRandomUniformBmodel/random_affine_transform_params/random_uniform/shape:output:0*
T0*#
_output_shapes
:         *
dtype0н
*model/random_affine_transform_params/RoundRoundJmodel/random_affine_transform_params/random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:         o
*model/random_affine_transform_params/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @┬
(model/random_affine_transform_params/mulMul.model/random_affine_transform_params/Round:y:03model/random_affine_transform_params/mul/y:output:0*
T0*#
_output_shapes
:         o
*model/random_affine_transform_params/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?└
(model/random_affine_transform_params/subSub,model/random_affine_transform_params/mul:z:03model/random_affine_transform_params/sub/y:output:0*
T0*#
_output_shapes
:         о
;model/random_affine_transform_params/random_uniform_1/shapePack;model/random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:~
9model/random_affine_transform_params/random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *█I└~
9model/random_affine_transform_params/random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *█I@╒
Cmodel/random_affine_transform_params/random_uniform_1/RandomUniformRandomUniformDmodel/random_affine_transform_params/random_uniform_1/shape:output:0*
T0*#
_output_shapes
:         *
dtype0щ
9model/random_affine_transform_params/random_uniform_1/subSubBmodel/random_affine_transform_params/random_uniform_1/max:output:0Bmodel/random_affine_transform_params/random_uniform_1/min:output:0*
T0*
_output_shapes
: √
9model/random_affine_transform_params/random_uniform_1/mulMulLmodel/random_affine_transform_params/random_uniform_1/RandomUniform:output:0=model/random_affine_transform_params/random_uniform_1/sub:z:0*
T0*#
_output_shapes
:         я
5model/random_affine_transform_params/random_uniform_1AddV2=model/random_affine_transform_params/random_uniform_1/mul:z:0Bmodel/random_affine_transform_params/random_uniform_1/min:output:0*
T0*#
_output_shapes
:         Ш
(model/random_affine_transform_params/CosCos9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         Ш
(model/random_affine_transform_params/SinSin9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         Л
(model/random_affine_transform_params/NegNeg,model/random_affine_transform_params/Sin:y:0*
T0*#
_output_shapes
:         ╗
*model/random_affine_transform_params/mul_1Mul,model/random_affine_transform_params/Neg:y:0,model/random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:         Ъ
*model/random_affine_transform_params/Sin_1Sin9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         Ъ
*model/random_affine_transform_params/Cos_1Cos9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         ╜
*model/random_affine_transform_params/mul_2Mul.model/random_affine_transform_params/Cos_1:y:0,model/random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:         ╬
-model/random_affine_transform_params/packed/0Pack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/mul_1:z:0*
N*
T0*'
_output_shapes
:         ╨
-model/random_affine_transform_params/packed/1Pack.model/random_affine_transform_params/Sin_1:y:0.model/random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:         т
+model/random_affine_transform_params/packedPack6model/random_affine_transform_params/packed/0:output:06model/random_affine_transform_params/packed/1:output:0*
N*
T0*+
_output_shapes
:         И
3model/random_affine_transform_params/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          х
.model/random_affine_transform_params/transpose	Transpose4model/random_affine_transform_params/packed:output:0<model/random_affine_transform_params/transpose/perm:output:0*
T0*+
_output_shapes
:         ╨
/model/random_affine_transform_params/packed_1/0Pack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/Sin_1:y:0*
N*
T0*'
_output_shapes
:         ╥
/model/random_affine_transform_params/packed_1/1Pack.model/random_affine_transform_params/mul_1:z:0.model/random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:         ш
-model/random_affine_transform_params/packed_1Pack8model/random_affine_transform_params/packed_1/0:output:08model/random_affine_transform_params/packed_1/1:output:0*
N*
T0*+
_output_shapes
:         К
5model/random_affine_transform_params/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ы
0model/random_affine_transform_params/transpose_1	Transpose6model/random_affine_transform_params/packed_1:output:0>model/random_affine_transform_params/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Г
*model/random_affine_transform_params/ConstConst*
_output_shapes

:*
dtype0*!
valueB"є5Cє5CЕ
,model/random_affine_transform_params/Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   C▌
+model/random_affine_transform_params/MatMulBatchMatMulV22model/random_affine_transform_params/transpose:y:05model/random_affine_transform_params/Const_1:output:0*
T0*+
_output_shapes
:         ╥
*model/random_affine_transform_params/sub_1Sub3model/random_affine_transform_params/Const:output:04model/random_affine_transform_params/MatMul:output:0*
T0*+
_output_shapes
:         ┌
-model/random_affine_transform_params/MatMul_1BatchMatMulV24model/random_affine_transform_params/transpose_1:y:0.model/random_affine_transform_params/sub_1:z:0*
T0*+
_output_shapes
:         П
:model/random_affine_transform_params/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            С
<model/random_affine_transform_params/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          С
<model/random_affine_transform_params/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ─
4model/random_affine_transform_params/strided_slice_1StridedSlice6model/random_affine_transform_params/MatMul_1:output:0Cmodel/random_affine_transform_params/strided_slice_1/stack:output:0Emodel/random_affine_transform_params/strided_slice_1/stack_1:output:0Emodel/random_affine_transform_params/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЮ
*model/random_affine_transform_params/Neg_1Neg=model/random_affine_transform_params/strided_slice_1:output:0*
T0*#
_output_shapes
:         П
:model/random_affine_transform_params/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           С
<model/random_affine_transform_params/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          С
<model/random_affine_transform_params/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ─
4model/random_affine_transform_params/strided_slice_2StridedSlice6model/random_affine_transform_params/MatMul_1:output:0Cmodel/random_affine_transform_params/strided_slice_2/stack:output:0Emodel/random_affine_transform_params/strided_slice_2/stack_1:output:0Emodel/random_affine_transform_params/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЮ
*model/random_affine_transform_params/Neg_2Neg=model/random_affine_transform_params/strided_slice_2:output:0*
T0*#
_output_shapes
:         П
*model/random_affine_transform_params/Neg_3Neg.model/random_affine_transform_params/Neg_1:y:0*
T0*#
_output_shapes
:         ╜
*model/random_affine_transform_params/mul_3Mul.model/random_affine_transform_params/Neg_3:y:0,model/random_affine_transform_params/Cos:y:0*
T0*#
_output_shapes
:         ┐
*model/random_affine_transform_params/mul_4Mul.model/random_affine_transform_params/Neg_2:y:0.model/random_affine_transform_params/mul_1:z:0*
T0*#
_output_shapes
:         ┐
*model/random_affine_transform_params/sub_2Sub.model/random_affine_transform_params/mul_3:z:0.model/random_affine_transform_params/mul_4:z:0*
T0*#
_output_shapes
:         П
*model/random_affine_transform_params/Neg_4Neg.model/random_affine_transform_params/Neg_2:y:0*
T0*#
_output_shapes
:         ┐
*model/random_affine_transform_params/mul_5Mul.model/random_affine_transform_params/Neg_4:y:0.model/random_affine_transform_params/mul_2:z:0*
T0*#
_output_shapes
:         ┐
*model/random_affine_transform_params/mul_6Mul.model/random_affine_transform_params/Neg_1:y:0.model/random_affine_transform_params/Sin_1:y:0*
T0*#
_output_shapes
:         ┐
*model/random_affine_transform_params/sub_3Sub.model/random_affine_transform_params/mul_5:z:0.model/random_affine_transform_params/mul_6:z:0*
T0*#
_output_shapes
:         Л
8model/random_affine_transform_params/zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         т
2model/random_affine_transform_params/zeros/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Amodel/random_affine_transform_params/zeros/Reshape/shape:output:0*
T0*
_output_shapes
:u
0model/random_affine_transform_params/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╪
*model/random_affine_transform_params/zerosFill;model/random_affine_transform_params/zeros/Reshape:output:09model/random_affine_transform_params/zeros/Const:output:0*
T0*#
_output_shapes
:         Н
:model/random_affine_transform_params/zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ц
4model/random_affine_transform_params/zeros_1/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Cmodel/random_affine_transform_params/zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:w
2model/random_affine_transform_params/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▐
,model/random_affine_transform_params/zeros_1Fill=model/random_affine_transform_params/zeros_1/Reshape:output:0;model/random_affine_transform_params/zeros_1/Const:output:0*
T0*#
_output_shapes
:         Г
*model/random_affine_transform_params/stackPack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/mul_1:z:0.model/random_affine_transform_params/sub_2:z:0.model/random_affine_transform_params/Sin_1:y:0.model/random_affine_transform_params/mul_2:z:0.model/random_affine_transform_params/sub_3:z:03model/random_affine_transform_params/zeros:output:05model/random_affine_transform_params/zeros_1:output:0*
N*
T0*'
_output_shapes
:         *

axisН
:model/random_affine_transform_params/zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ц
4model/random_affine_transform_params/zeros_2/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Cmodel/random_affine_transform_params/zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:w
2model/random_affine_transform_params/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▐
,model/random_affine_transform_params/zeros_2Fill=model/random_affine_transform_params/zeros_2/Reshape:output:0;model/random_affine_transform_params/zeros_2/Const:output:0*
T0*#
_output_shapes
:         Н
:model/random_affine_transform_params/zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ц
4model/random_affine_transform_params/zeros_3/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Cmodel/random_affine_transform_params/zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:w
2model/random_affine_transform_params/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▐
,model/random_affine_transform_params/zeros_3Fill=model/random_affine_transform_params/zeros_3/Reshape:output:0;model/random_affine_transform_params/zeros_3/Const:output:0*
T0*#
_output_shapes
:         З
,model/random_affine_transform_params/stack_1Pack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/Sin_1:y:0.model/random_affine_transform_params/Neg_1:y:0.model/random_affine_transform_params/mul_1:z:0.model/random_affine_transform_params/mul_2:z:0.model/random_affine_transform_params/Neg_2:y:05model/random_affine_transform_params/zeros_2:output:05model/random_affine_transform_params/zeros_3:output:0*
N*
T0*'
_output_shapes
:         *

axisЯ
Nmodel/image_projective_transform_layer/ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"v  v  С
Lmodel/image_projective_transform_layer/ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    м
Amodel/image_projective_transform_layer/ImageProjectiveTransformV3ImageProjectiveTransformV3input_15model/random_affine_transform_params/stack_1:output:0Wmodel/image_projective_transform_layer/ImageProjectiveTransformV3/output_shape:output:0Umodel/image_projective_transform_layer/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         ЎЎ*
dtype0*
interpolation
BILINEARФ
#model/tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ╟
model/tf.compat.v1.pad/PadPad*model/tf.compat.v1.transpose/transpose:y:0,model/tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+
                           к
-model/ResNet/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
model/ResNet/conv2d_1/Conv2D_1Conv2DVmodel/image_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:05model/ResNet/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
а
.model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╟
model/ResNet/conv2d_1/BiasAdd_1BiasAdd'model/ResNet/conv2d_1/Conv2D_1:output:06model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎк
3model/ResNet/batch_normalization_1/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0м
3model/ResNet/batch_normalization_1/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0╠
Dmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╨
Fmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0С
5model/ResNet/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_1/BiasAdd_1:output:0;model/ResNet/batch_normalization_1/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_1/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( Я
 model/ResNet/activation_1/Relu_1Relu9model/ResNet/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎж
+model/ResNet/conv2d/Conv2D_1/ReadVariableOpReadVariableOp2model_resnet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ч
model/ResNet/conv2d/Conv2D_1Conv2DVmodel/image_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:03model/ResNet/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
Ь
,model/ResNet/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp3model_resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┴
model/ResNet/conv2d/BiasAdd_1BiasAdd%model/ResNet/conv2d/Conv2D_1:output:04model/ResNet/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎк
-model/ResNet/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0є
model/ResNet/conv2d_2/Conv2D_1Conv2D.model/ResNet/activation_1/Relu_1:activations:05model/ResNet/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
а
.model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╟
model/ResNet/conv2d_2/BiasAdd_1BiasAdd'model/ResNet/conv2d_2/Conv2D_1:output:06model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎж
1model/ResNet/batch_normalization/ReadVariableOp_2ReadVariableOp8model_resnet_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0и
1model/ResNet/batch_normalization/ReadVariableOp_3ReadVariableOp:model_resnet_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
Bmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpImodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╠
Dmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpKmodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Е
3model/ResNet/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3&model/ResNet/conv2d/BiasAdd_1:output:09model/ResNet/batch_normalization/ReadVariableOp_2:value:09model/ResNet/batch_normalization/ReadVariableOp_3:value:0Jmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( к
3model/ResNet/batch_normalization_2/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3model/ResNet/batch_normalization_2/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╠
Dmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╨
Fmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0С
5model/ResNet/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_2/BiasAdd_1:output:0;model/ResNet/batch_normalization_2/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_2/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( Я
 model/ResNet/activation_2/Relu_1Relu9model/ResNet/batch_normalization_2/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎЫ
model/ResNet/activation/Relu_1Relu7model/ResNet/batch_normalization/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ╣
model/ResNet/add/add_1AddV2.model/ResNet/activation_2/Relu_1:activations:0,model/ResNet/activation/Relu_1:activations:0*
T0*1
_output_shapes
:         ЎЎк
3model/ResNet/batch_normalization_3/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0м
3model/ResNet/batch_normalization_3/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0╠
Dmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╨
Fmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Г
5model/ResNet/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3model/ResNet/add/add_1:z:0;model/ResNet/batch_normalization_3/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_3/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( Я
 model/ResNet/activation_3/Relu_1Relu9model/ResNet/batch_normalization_3/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎк
-model/ResNet/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0є
model/ResNet/conv2d_4/Conv2D_1Conv2D.model/ResNet/activation_3/Relu_1:activations:05model/ResNet/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
а
.model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╟
model/ResNet/conv2d_4/BiasAdd_1BiasAdd'model/ResNet/conv2d_4/Conv2D_1:output:06model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ к
3model/ResNet/batch_normalization_5/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_5/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
Dmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╨
Fmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0С
5model/ResNet/batch_normalization_5/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_4/BiasAdd_1:output:0;model/ResNet/batch_normalization_5/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_5/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( Я
 model/ResNet/activation_5/Relu_1Relu9model/ResNet/batch_normalization_5/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ к
-model/ResNet/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0є
model/ResNet/conv2d_3/Conv2D_1Conv2D.model/ResNet/activation_3/Relu_1:activations:05model/ResNet/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
а
.model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╟
model/ResNet/conv2d_3/BiasAdd_1BiasAdd'model/ResNet/conv2d_3/Conv2D_1:output:06model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ к
-model/ResNet/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0є
model/ResNet/conv2d_5/Conv2D_1Conv2D.model/ResNet/activation_5/Relu_1:activations:05model/ResNet/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
а
.model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╟
model/ResNet/conv2d_5/BiasAdd_1BiasAdd'model/ResNet/conv2d_5/Conv2D_1:output:06model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ к
3model/ResNet/batch_normalization_4/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_4/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
Dmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╨
Fmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0С
5model/ResNet/batch_normalization_4/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_3/BiasAdd_1:output:0;model/ResNet/batch_normalization_4/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_4/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( к
3model/ResNet/batch_normalization_6/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_6/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
Dmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╨
Fmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0С
5model/ResNet/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_5/BiasAdd_1:output:0;model/ResNet/batch_normalization_6/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_6/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( Я
 model/ResNet/activation_6/Relu_1Relu9model/ResNet/batch_normalization_6/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ Я
 model/ResNet/activation_4/Relu_1Relu9model/ResNet/batch_normalization_4/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ ╜
model/ResNet/add_1/add_1AddV2.model/ResNet/activation_6/Relu_1:activations:0.model/ResNet/activation_4/Relu_1:activations:0*
T0*1
_output_shapes
:         ЎЎ к
3model/ResNet/batch_normalization_7/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0м
3model/ResNet/batch_normalization_7/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
Dmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╨
Fmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Е
5model/ResNet/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3model/ResNet/add_1/add_1:z:0;model/ResNet/batch_normalization_7/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_7/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( Я
 model/ResNet/activation_7/Relu_1Relu9model/ResNet/batch_normalization_7/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ к
-model/ResNet/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0є
model/ResNet/conv2d_6/Conv2D_1Conv2D.model/ResNet/activation_7/Relu_1:activations:05model/ResNet/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ
*
paddingSAME*
strides
а
.model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╟
model/ResNet/conv2d_6/BiasAdd_1BiasAdd'model/ResNet/conv2d_6/Conv2D_1:output:06model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ
Р
model/ResNet/conv2d_6/Softmax_1Softmax(model/ResNet/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:         ЎЎ
б
Pmodel/image_projective_transform_layer_1/ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      У
Nmodel/image_projective_transform_layer_1/ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ╥
Cmodel/image_projective_transform_layer_1/ImageProjectiveTransformV3ImageProjectiveTransformV3)model/ResNet/conv2d_6/Softmax_1:softmax:03model/random_affine_transform_params/stack:output:0Ymodel/image_projective_transform_layer_1/ImageProjectiveTransformV3/output_shape:output:0Wmodel/image_projective_transform_layer_1/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         АА
*
dtype0*
interpolation
BILINEARЖ
-model/tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Г
(model/tf.compat.v1.transpose_1/transpose	TransposeXmodel/image_projective_transform_layer_1/ImageProjectiveTransformV3:transformed_images:06model/tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА         
ь
#model/tf.compat.v1.nn.conv2d/Conv2DConv2D#model/tf.compat.v1.pad/Pad:output:0,model/tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"
                  
*
paddingVALID*
strides
Л
2model/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              Н
4model/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"              Н
4model/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Х
,model/tf.__operators__.getitem/strided_sliceStridedSlice,model/tf.compat.v1.nn.conv2d/Conv2D:output:0;model/tf.__operators__.getitem/strided_slice/stack:output:0=model/tf.__operators__.getitem/strided_slice/stack_1:output:0=model/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask	*
end_mask	*
shrink_axis_maskН
"model/tf.compat.v1.squeeze/SqueezeSqueeze5model/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:

d
model/tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АAи
model/tf.math.truediv/truedivRealDiv+model/tf.compat.v1.squeeze/Squeeze:output:0(model/tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:

f
!model/tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCв
model/tf.math.truediv_1/truedivRealDiv!model/tf.math.truediv/truediv:z:0*model/tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:

f
!model/tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCд
model/tf.math.truediv_2/truedivRealDiv#model/tf.math.truediv_1/truediv:z:0*model/tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:

~
-model/tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ╗
(model/tf.compat.v1.transpose_2/transpose	Transpose#model/tf.math.truediv_2/truediv:z:06model/tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:

е
 model/tf.__operators__.add/AddV2AddV2#model/tf.math.truediv_2/truediv:z:0,model/tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:

f
!model/tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @е
model/tf.math.truediv_3/truedivRealDiv$model/tf.__operators__.add/AddV2:z:0*model/tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:

c
model/tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3в
"model/tf.__operators__.add_2/AddV2AddV2#model/tf.math.truediv_3/truediv:z:0'model/tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:

y
.model/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         ╗
model/tf.math.reduce_sum/SumSum#model/tf.math.truediv_3/truediv:z:07model/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims({
0model/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        ┐
model/tf.math.reduce_sum_1/SumSum#model/tf.math.truediv_3/truediv:z:09model/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(Ъ
model/tf.math.multiply/MulMul%model/tf.math.reduce_sum/Sum:output:0'model/tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:

c
model/tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Э
"model/tf.__operators__.add_1/AddV2AddV2model/tf.math.multiply/Mul:z:0'model/tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:

г
model/tf.math.truediv_4/truedivRealDiv&model/tf.__operators__.add_1/AddV2:z:0&model/tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:

j
model/tf.math.log/LogLog#model/tf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:

М
model/tf.math.multiply_1/MulMul#model/tf.math.truediv_3/truediv:z:0model/tf.math.log/Log:y:0*
T0*
_output_shapes

:

Б
0model/tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"    ■   г
model/tf.math.reduce_sum_2/SumSum model/tf.math.multiply_1/Mul:z:09model/tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*
_output_shapes
: `
model/tf.math.reduce_mean/RankConst*
_output_shapes
: *
dtype0*
value	B : g
%model/tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : g
%model/tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :╦
model/tf.math.reduce_mean/rangeRange.model/tf.math.reduce_mean/range/start:output:0'model/tf.math.reduce_mean/Rank:output:0.model/tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: Ъ
model/tf.math.reduce_mean/MeanMean'model/tf.math.reduce_sum_2/Sum:output:0(model/tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: Р
IdentityIdentity'model/ResNet/conv2d_6/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+                           
Ё)
NoOpNoOpA^model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpC^model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1C^model/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpE^model/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_10^model/ResNet/batch_normalization/ReadVariableOp2^model/ResNet/batch_normalization/ReadVariableOp_12^model/ResNet/batch_normalization/ReadVariableOp_22^model/ResNet/batch_normalization/ReadVariableOp_3C^model/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_1/ReadVariableOp4^model/ResNet/batch_normalization_1/ReadVariableOp_14^model/ResNet/batch_normalization_1/ReadVariableOp_24^model/ResNet/batch_normalization_1/ReadVariableOp_3C^model/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_2/ReadVariableOp4^model/ResNet/batch_normalization_2/ReadVariableOp_14^model/ResNet/batch_normalization_2/ReadVariableOp_24^model/ResNet/batch_normalization_2/ReadVariableOp_3C^model/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_3/ReadVariableOp4^model/ResNet/batch_normalization_3/ReadVariableOp_14^model/ResNet/batch_normalization_3/ReadVariableOp_24^model/ResNet/batch_normalization_3/ReadVariableOp_3C^model/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_4/ReadVariableOp4^model/ResNet/batch_normalization_4/ReadVariableOp_14^model/ResNet/batch_normalization_4/ReadVariableOp_24^model/ResNet/batch_normalization_4/ReadVariableOp_3C^model/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_5/ReadVariableOp4^model/ResNet/batch_normalization_5/ReadVariableOp_14^model/ResNet/batch_normalization_5/ReadVariableOp_24^model/ResNet/batch_normalization_5/ReadVariableOp_3C^model/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_6/ReadVariableOp4^model/ResNet/batch_normalization_6/ReadVariableOp_14^model/ResNet/batch_normalization_6/ReadVariableOp_24^model/ResNet/batch_normalization_6/ReadVariableOp_3C^model/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_7/ReadVariableOp4^model/ResNet/batch_normalization_7/ReadVariableOp_14^model/ResNet/batch_normalization_7/ReadVariableOp_24^model/ResNet/batch_normalization_7/ReadVariableOp_3+^model/ResNet/conv2d/BiasAdd/ReadVariableOp-^model/ResNet/conv2d/BiasAdd_1/ReadVariableOp*^model/ResNet/conv2d/Conv2D/ReadVariableOp,^model/ResNet/conv2d/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_1/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_1/Conv2D/ReadVariableOp.^model/ResNet/conv2d_1/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_2/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_2/Conv2D/ReadVariableOp.^model/ResNet/conv2d_2/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_3/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_3/Conv2D/ReadVariableOp.^model/ResNet/conv2d_3/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_4/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_4/Conv2D/ReadVariableOp.^model/ResNet/conv2d_4/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_5/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_5/Conv2D/ReadVariableOp.^model/ResNet/conv2d_5/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_6/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_6/Conv2D/ReadVariableOp.^model/ResNet/conv2d_6/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Д
@model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp@model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp2И
Bmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Bmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_12И
Bmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpBmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2М
Dmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Dmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12b
/model/ResNet/batch_normalization/ReadVariableOp/model/ResNet/batch_normalization/ReadVariableOp2f
1model/ResNet/batch_normalization/ReadVariableOp_11model/ResNet/batch_normalization/ReadVariableOp_12f
1model/ResNet/batch_normalization/ReadVariableOp_21model/ResNet/batch_normalization/ReadVariableOp_22f
1model/ResNet/batch_normalization/ReadVariableOp_31model/ResNet/batch_normalization/ReadVariableOp_32И
Bmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpBmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2М
Dmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Dmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12М
Dmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpDmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2Р
Fmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Fmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12f
1model/ResNet/batch_normalization_1/ReadVariableOp1model/ResNet/batch_normalization_1/ReadVariableOp2j
3model/ResNet/batch_normalization_1/ReadVariableOp_13model/ResNet/batch_normalization_1/ReadVariableOp_12j
3model/ResNet/batch_normalization_1/ReadVariableOp_23model/ResNet/batch_normalization_1/ReadVariableOp_22j
3model/ResNet/batch_normalization_1/ReadVariableOp_33model/ResNet/batch_normalization_1/ReadVariableOp_32И
Bmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12М
Dmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpDmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp2Р
Fmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1Fmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12f
1model/ResNet/batch_normalization_2/ReadVariableOp1model/ResNet/batch_normalization_2/ReadVariableOp2j
3model/ResNet/batch_normalization_2/ReadVariableOp_13model/ResNet/batch_normalization_2/ReadVariableOp_12j
3model/ResNet/batch_normalization_2/ReadVariableOp_23model/ResNet/batch_normalization_2/ReadVariableOp_22j
3model/ResNet/batch_normalization_2/ReadVariableOp_33model/ResNet/batch_normalization_2/ReadVariableOp_32И
Bmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2М
Dmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12М
Dmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpDmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp2Р
Fmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1Fmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12f
1model/ResNet/batch_normalization_3/ReadVariableOp1model/ResNet/batch_normalization_3/ReadVariableOp2j
3model/ResNet/batch_normalization_3/ReadVariableOp_13model/ResNet/batch_normalization_3/ReadVariableOp_12j
3model/ResNet/batch_normalization_3/ReadVariableOp_23model/ResNet/batch_normalization_3/ReadVariableOp_22j
3model/ResNet/batch_normalization_3/ReadVariableOp_33model/ResNet/batch_normalization_3/ReadVariableOp_32И
Bmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2М
Dmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12М
Dmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpDmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp2Р
Fmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1Fmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_12f
1model/ResNet/batch_normalization_4/ReadVariableOp1model/ResNet/batch_normalization_4/ReadVariableOp2j
3model/ResNet/batch_normalization_4/ReadVariableOp_13model/ResNet/batch_normalization_4/ReadVariableOp_12j
3model/ResNet/batch_normalization_4/ReadVariableOp_23model/ResNet/batch_normalization_4/ReadVariableOp_22j
3model/ResNet/batch_normalization_4/ReadVariableOp_33model/ResNet/batch_normalization_4/ReadVariableOp_32И
Bmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12М
Dmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpDmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp2Р
Fmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1Fmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_12f
1model/ResNet/batch_normalization_5/ReadVariableOp1model/ResNet/batch_normalization_5/ReadVariableOp2j
3model/ResNet/batch_normalization_5/ReadVariableOp_13model/ResNet/batch_normalization_5/ReadVariableOp_12j
3model/ResNet/batch_normalization_5/ReadVariableOp_23model/ResNet/batch_normalization_5/ReadVariableOp_22j
3model/ResNet/batch_normalization_5/ReadVariableOp_33model/ResNet/batch_normalization_5/ReadVariableOp_32И
Bmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2М
Dmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12М
Dmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpDmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp2Р
Fmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1Fmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12f
1model/ResNet/batch_normalization_6/ReadVariableOp1model/ResNet/batch_normalization_6/ReadVariableOp2j
3model/ResNet/batch_normalization_6/ReadVariableOp_13model/ResNet/batch_normalization_6/ReadVariableOp_12j
3model/ResNet/batch_normalization_6/ReadVariableOp_23model/ResNet/batch_normalization_6/ReadVariableOp_22j
3model/ResNet/batch_normalization_6/ReadVariableOp_33model/ResNet/batch_normalization_6/ReadVariableOp_32И
Bmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12М
Dmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpDmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp2Р
Fmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1Fmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12f
1model/ResNet/batch_normalization_7/ReadVariableOp1model/ResNet/batch_normalization_7/ReadVariableOp2j
3model/ResNet/batch_normalization_7/ReadVariableOp_13model/ResNet/batch_normalization_7/ReadVariableOp_12j
3model/ResNet/batch_normalization_7/ReadVariableOp_23model/ResNet/batch_normalization_7/ReadVariableOp_22j
3model/ResNet/batch_normalization_7/ReadVariableOp_33model/ResNet/batch_normalization_7/ReadVariableOp_32X
*model/ResNet/conv2d/BiasAdd/ReadVariableOp*model/ResNet/conv2d/BiasAdd/ReadVariableOp2\
,model/ResNet/conv2d/BiasAdd_1/ReadVariableOp,model/ResNet/conv2d/BiasAdd_1/ReadVariableOp2V
)model/ResNet/conv2d/Conv2D/ReadVariableOp)model/ResNet/conv2d/Conv2D/ReadVariableOp2Z
+model/ResNet/conv2d/Conv2D_1/ReadVariableOp+model/ResNet/conv2d/Conv2D_1/ReadVariableOp2\
,model/ResNet/conv2d_1/BiasAdd/ReadVariableOp,model/ResNet/conv2d_1/BiasAdd/ReadVariableOp2`
.model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOp.model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOp2Z
+model/ResNet/conv2d_1/Conv2D/ReadVariableOp+model/ResNet/conv2d_1/Conv2D/ReadVariableOp2^
-model/ResNet/conv2d_1/Conv2D_1/ReadVariableOp-model/ResNet/conv2d_1/Conv2D_1/ReadVariableOp2\
,model/ResNet/conv2d_2/BiasAdd/ReadVariableOp,model/ResNet/conv2d_2/BiasAdd/ReadVariableOp2`
.model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOp.model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOp2Z
+model/ResNet/conv2d_2/Conv2D/ReadVariableOp+model/ResNet/conv2d_2/Conv2D/ReadVariableOp2^
-model/ResNet/conv2d_2/Conv2D_1/ReadVariableOp-model/ResNet/conv2d_2/Conv2D_1/ReadVariableOp2\
,model/ResNet/conv2d_3/BiasAdd/ReadVariableOp,model/ResNet/conv2d_3/BiasAdd/ReadVariableOp2`
.model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOp.model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOp2Z
+model/ResNet/conv2d_3/Conv2D/ReadVariableOp+model/ResNet/conv2d_3/Conv2D/ReadVariableOp2^
-model/ResNet/conv2d_3/Conv2D_1/ReadVariableOp-model/ResNet/conv2d_3/Conv2D_1/ReadVariableOp2\
,model/ResNet/conv2d_4/BiasAdd/ReadVariableOp,model/ResNet/conv2d_4/BiasAdd/ReadVariableOp2`
.model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOp.model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOp2Z
+model/ResNet/conv2d_4/Conv2D/ReadVariableOp+model/ResNet/conv2d_4/Conv2D/ReadVariableOp2^
-model/ResNet/conv2d_4/Conv2D_1/ReadVariableOp-model/ResNet/conv2d_4/Conv2D_1/ReadVariableOp2\
,model/ResNet/conv2d_5/BiasAdd/ReadVariableOp,model/ResNet/conv2d_5/BiasAdd/ReadVariableOp2`
.model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOp.model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOp2Z
+model/ResNet/conv2d_5/Conv2D/ReadVariableOp+model/ResNet/conv2d_5/Conv2D/ReadVariableOp2^
-model/ResNet/conv2d_5/Conv2D_1/ReadVariableOp-model/ResNet/conv2d_5/Conv2D_1/ReadVariableOp2\
,model/ResNet/conv2d_6/BiasAdd/ReadVariableOp,model/ResNet/conv2d_6/BiasAdd/ReadVariableOp2`
.model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOp.model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOp2Z
+model/ResNet/conv2d_6/Conv2D/ReadVariableOp+model/ResNet/conv2d_6/Conv2D/ReadVariableOp2^
-model/ResNet/conv2d_6/Conv2D_1/ReadVariableOp-model/ResNet/conv2d_6/Conv2D_1/ReadVariableOp:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_1
З
┴
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387162

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ш
╙

'__inference_model_layer_call_fn_1390110

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╚
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+                           
: *@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1389233Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387259

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
│
c
G__inference_activation_layer_call_and_return_conditional_losses_1387466

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╡
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1392228

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╦
Ы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387003

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392208

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ф	
╨
5__inference_batch_normalization_layer_call_fn_1391802

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387003Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш	
╥
7__inference_batch_normalization_4_layer_call_fn_1392159

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387259Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ш	
╥
7__inference_batch_normalization_2_layer_call_fn_1391740

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386939Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╤
j
@__inference_add_layer_call_and_return_conditional_losses_1387474

inputs
inputs_1
identityj
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+                           i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+                           :+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
╘

'__inference_model_layer_call_fn_1388911
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+                           
: *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1388815Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_1
Б
№
C__inference_conv2d_layer_call_and_return_conditional_losses_1387414

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16793
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
л
J
"__inference__update_step_xla_16773
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╡
e
I__inference_activation_6_layer_call_and_return_conditional_losses_1387579

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ш	
╥
7__inference_batch_normalization_1_layer_call_fn_1391630

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386875Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╗
Q
%__inference_add_layer_call_fn_1391877
inputs_0
inputs_1
identity╫
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_1387474z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+                           :+                           :k g
A
_output_shapes/
-:+                           
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+                           
"
_user_specified_name
inputs_1
л
J
"__inference__update_step_xla_16858
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ф
J
.__inference_activation_5_layer_call_fn_1392041

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1387522z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Г
■
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1387550

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387098

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16843
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╢
Э
(__inference_conv2d_layer_call_fn_1391717

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1387414Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_1_layer_call_fn_1391643

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386906Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
НE
З
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1388695
inp
identity

identity_1И8
ShapeShapeinp*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
random_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:З
random_uniform/RandomUniformRandomUniformrandom_uniform/shape:output:0*
T0*#
_output_shapes
:         *
dtype0c
RoundRound%random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:         J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @S
mulMul	Round:y:0mul/y:output:0*
T0*#
_output_shapes
:         J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Q
subSubmul:z:0sub/y:output:0*
T0*#
_output_shapes
:         d
random_uniform_1/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *█I└Y
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *█I@Л
random_uniform_1/RandomUniformRandomUniformrandom_uniform_1/shape:output:0*
T0*#
_output_shapes
:         *
dtype0z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: М
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:         А
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:         N
CosCosrandom_uniform_1:z:0*
T0*#
_output_shapes
:         N
SinSinrandom_uniform_1:z:0*
T0*#
_output_shapes
:         A
NegNegSin:y:0*
T0*#
_output_shapes
:         L
mul_1MulNeg:y:0sub:z:0*
T0*#
_output_shapes
:         P
Sin_1Sinrandom_uniform_1:z:0*
T0*#
_output_shapes
:         P
Cos_1Cosrandom_uniform_1:z:0*
T0*#
_output_shapes
:         N
mul_2Mul	Cos_1:y:0sub:z:0*
T0*#
_output_shapes
:         _
packed/0PackCos:y:0	mul_1:z:0*
N*
T0*'
_output_shapes
:         a
packed/1Pack	Sin_1:y:0	mul_2:z:0*
N*
T0*'
_output_shapes
:         s
packedPackpacked/0:output:0packed/1:output:0*
N*
T0*+
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:         a

packed_1/0PackCos:y:0	Sin_1:y:0*
N*
T0*'
_output_shapes
:         c

packed_1/1Pack	mul_1:z:0	mul_2:z:0*
N*
T0*'
_output_shapes
:         y
packed_1Packpacked_1/0:output:0packed_1/1:output:0*
N*
T0*+
_output_shapes
:         e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_1	Transposepacked_1:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         ^
ConstConst*
_output_shapes

:*
dtype0*!
valueB"є5Cє5C`
Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   Cn
MatMulBatchMatMulV2transpose:y:0Const_1:output:0*
T0*+
_output_shapes
:         c
sub_1SubConst:output:0MatMul:output:0*
T0*+
_output_shapes
:         k
MatMul_1BatchMatMulV2transpose_1:y:0	sub_1:z:0*
T0*+
_output_shapes
:         j
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Л
strided_slice_1StridedSliceMatMul_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskT
Neg_1Negstrided_slice_1:output:0*
T0*#
_output_shapes
:         j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Л
strided_slice_2StridedSliceMatMul_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskT
Neg_2Negstrided_slice_2:output:0*
T0*#
_output_shapes
:         E
Neg_3Neg	Neg_1:y:0*
T0*#
_output_shapes
:         N
mul_3Mul	Neg_3:y:0Cos:y:0*
T0*#
_output_shapes
:         P
mul_4Mul	Neg_2:y:0	mul_1:z:0*
T0*#
_output_shapes
:         P
sub_2Sub	mul_3:z:0	mul_4:z:0*
T0*#
_output_shapes
:         E
Neg_4Neg	Neg_2:y:0*
T0*#
_output_shapes
:         P
mul_5Mul	Neg_4:y:0	mul_2:z:0*
T0*#
_output_shapes
:         P
mul_6Mul	Neg_1:y:0	Sin_1:y:0*
T0*#
_output_shapes
:         P
sub_3Sub	mul_5:z:0	mul_6:z:0*
T0*#
_output_shapes
:         f
zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         s
zeros/ReshapeReshapestrided_slice:output:0zeros/Reshape/shape:output:0*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    i
zerosFillzeros/Reshape:output:0zeros/Const:output:0*
T0*#
_output_shapes
:         h
zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         w
zeros_1/ReshapeReshapestrided_slice:output:0zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    o
zeros_1Fillzeros_1/Reshape:output:0zeros_1/Const:output:0*
T0*#
_output_shapes
:         ╢
stackPackCos:y:0	mul_1:z:0	sub_2:z:0	Sin_1:y:0	mul_2:z:0	sub_3:z:0zeros:output:0zeros_1:output:0*
N*
T0*'
_output_shapes
:         *

axish
zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         w
zeros_2/ReshapeReshapestrided_slice:output:0zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    o
zeros_2Fillzeros_2/Reshape:output:0zeros_2/Const:output:0*
T0*#
_output_shapes
:         h
zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         w
zeros_3/ReshapeReshapestrided_slice:output:0zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    o
zeros_3Fillzeros_3/Reshape:output:0zeros_3/Const:output:0*
T0*#
_output_shapes
:         ║
stack_1PackCos:y:0	Sin_1:y:0	Neg_1:y:0	mul_1:z:0	mul_2:z:0	Neg_2:y:0zeros_2:output:0zeros_3:output:0*
N*
T0*'
_output_shapes
:         *

axisX
IdentityIdentitystack_1:output:0*
T0*'
_output_shapes
:         X

Identity_1Identitystack:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :f b
A
_output_shapes/
-:+                           

_user_specified_nameinp
═
Э
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387131

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387067

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Е
┐
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387034

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16783
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╧
V
"__inference__update_step_xla_16818
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
═
Э
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387195

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
рь
╛L
B__inference_model_layer_call_and_return_conditional_losses_1390938

inputsH
.resnet_conv2d_1_conv2d_readvariableop_resource:=
/resnet_conv2d_1_biasadd_readvariableop_resource:B
4resnet_batch_normalization_1_readvariableop_resource:D
6resnet_batch_normalization_1_readvariableop_1_resource:S
Eresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:U
Gresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:F
,resnet_conv2d_conv2d_readvariableop_resource:;
-resnet_conv2d_biasadd_readvariableop_resource:H
.resnet_conv2d_2_conv2d_readvariableop_resource:=
/resnet_conv2d_2_biasadd_readvariableop_resource:@
2resnet_batch_normalization_readvariableop_resource:B
4resnet_batch_normalization_readvariableop_1_resource:Q
Cresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:S
Eresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:B
4resnet_batch_normalization_2_readvariableop_resource:D
6resnet_batch_normalization_2_readvariableop_1_resource:S
Eresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:U
Gresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:B
4resnet_batch_normalization_3_readvariableop_resource:D
6resnet_batch_normalization_3_readvariableop_1_resource:S
Eresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:U
Gresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:H
.resnet_conv2d_4_conv2d_readvariableop_resource: =
/resnet_conv2d_4_biasadd_readvariableop_resource: B
4resnet_batch_normalization_5_readvariableop_resource: D
6resnet_batch_normalization_5_readvariableop_1_resource: S
Eresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: H
.resnet_conv2d_3_conv2d_readvariableop_resource: =
/resnet_conv2d_3_biasadd_readvariableop_resource: H
.resnet_conv2d_5_conv2d_readvariableop_resource:  =
/resnet_conv2d_5_biasadd_readvariableop_resource: B
4resnet_batch_normalization_4_readvariableop_resource: D
6resnet_batch_normalization_4_readvariableop_1_resource: S
Eresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: B
4resnet_batch_normalization_6_readvariableop_resource: D
6resnet_batch_normalization_6_readvariableop_1_resource: S
Eresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: B
4resnet_batch_normalization_7_readvariableop_resource: D
6resnet_batch_normalization_7_readvariableop_1_resource: S
Eresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: H
.resnet_conv2d_6_conv2d_readvariableop_resource: 
=
/resnet_conv2d_6_biasadd_readvariableop_resource:

identity

identity_1Ив)ResNet/batch_normalization/AssignNewValueв+ResNet/batch_normalization/AssignNewValue_1в+ResNet/batch_normalization/AssignNewValue_2в+ResNet/batch_normalization/AssignNewValue_3в:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpв<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1в<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpв>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1в)ResNet/batch_normalization/ReadVariableOpв+ResNet/batch_normalization/ReadVariableOp_1в+ResNet/batch_normalization/ReadVariableOp_2в+ResNet/batch_normalization/ReadVariableOp_3в+ResNet/batch_normalization_1/AssignNewValueв-ResNet/batch_normalization_1/AssignNewValue_1в-ResNet/batch_normalization_1/AssignNewValue_2в-ResNet/batch_normalization_1/AssignNewValue_3в<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_1/ReadVariableOpв-ResNet/batch_normalization_1/ReadVariableOp_1в-ResNet/batch_normalization_1/ReadVariableOp_2в-ResNet/batch_normalization_1/ReadVariableOp_3в+ResNet/batch_normalization_2/AssignNewValueв-ResNet/batch_normalization_2/AssignNewValue_1в-ResNet/batch_normalization_2/AssignNewValue_2в-ResNet/batch_normalization_2/AssignNewValue_3в<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_2/ReadVariableOpв-ResNet/batch_normalization_2/ReadVariableOp_1в-ResNet/batch_normalization_2/ReadVariableOp_2в-ResNet/batch_normalization_2/ReadVariableOp_3в+ResNet/batch_normalization_3/AssignNewValueв-ResNet/batch_normalization_3/AssignNewValue_1в-ResNet/batch_normalization_3/AssignNewValue_2в-ResNet/batch_normalization_3/AssignNewValue_3в<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_3/ReadVariableOpв-ResNet/batch_normalization_3/ReadVariableOp_1в-ResNet/batch_normalization_3/ReadVariableOp_2в-ResNet/batch_normalization_3/ReadVariableOp_3в+ResNet/batch_normalization_4/AssignNewValueв-ResNet/batch_normalization_4/AssignNewValue_1в-ResNet/batch_normalization_4/AssignNewValue_2в-ResNet/batch_normalization_4/AssignNewValue_3в<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_4/ReadVariableOpв-ResNet/batch_normalization_4/ReadVariableOp_1в-ResNet/batch_normalization_4/ReadVariableOp_2в-ResNet/batch_normalization_4/ReadVariableOp_3в+ResNet/batch_normalization_5/AssignNewValueв-ResNet/batch_normalization_5/AssignNewValue_1в-ResNet/batch_normalization_5/AssignNewValue_2в-ResNet/batch_normalization_5/AssignNewValue_3в<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_5/ReadVariableOpв-ResNet/batch_normalization_5/ReadVariableOp_1в-ResNet/batch_normalization_5/ReadVariableOp_2в-ResNet/batch_normalization_5/ReadVariableOp_3в+ResNet/batch_normalization_6/AssignNewValueв-ResNet/batch_normalization_6/AssignNewValue_1в-ResNet/batch_normalization_6/AssignNewValue_2в-ResNet/batch_normalization_6/AssignNewValue_3в<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_6/ReadVariableOpв-ResNet/batch_normalization_6/ReadVariableOp_1в-ResNet/batch_normalization_6/ReadVariableOp_2в-ResNet/batch_normalization_6/ReadVariableOp_3в+ResNet/batch_normalization_7/AssignNewValueв-ResNet/batch_normalization_7/AssignNewValue_1в-ResNet/batch_normalization_7/AssignNewValue_2в-ResNet/batch_normalization_7/AssignNewValue_3в<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_7/ReadVariableOpв-ResNet/batch_normalization_7/ReadVariableOp_1в-ResNet/batch_normalization_7/ReadVariableOp_2в-ResNet/batch_normalization_7/ReadVariableOp_3в$ResNet/conv2d/BiasAdd/ReadVariableOpв&ResNet/conv2d/BiasAdd_1/ReadVariableOpв#ResNet/conv2d/Conv2D/ReadVariableOpв%ResNet/conv2d/Conv2D_1/ReadVariableOpв&ResNet/conv2d_1/BiasAdd/ReadVariableOpв(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_1/Conv2D/ReadVariableOpв'ResNet/conv2d_1/Conv2D_1/ReadVariableOpв&ResNet/conv2d_2/BiasAdd/ReadVariableOpв(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_2/Conv2D/ReadVariableOpв'ResNet/conv2d_2/Conv2D_1/ReadVariableOpв&ResNet/conv2d_3/BiasAdd/ReadVariableOpв(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_3/Conv2D/ReadVariableOpв'ResNet/conv2d_3/Conv2D_1/ReadVariableOpв&ResNet/conv2d_4/BiasAdd/ReadVariableOpв(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_4/Conv2D/ReadVariableOpв'ResNet/conv2d_4/Conv2D_1/ReadVariableOpв&ResNet/conv2d_5/BiasAdd/ReadVariableOpв(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_5/Conv2D/ReadVariableOpв'ResNet/conv2d_5/Conv2D_1/ReadVariableOpв&ResNet/conv2d_6/BiasAdd/ReadVariableOpв(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_6/Conv2D/ReadVariableOpв'ResNet/conv2d_6/Conv2D_1/ReadVariableOpЬ
%ResNet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╦
ResNet/conv2d_1/Conv2DConv2Dinputs-ResNet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Т
&ResNet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
ResNet/conv2d_1/BiasAddBiasAddResNet/conv2d_1/Conv2D:output:0.ResNet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           Ь
+ResNet/batch_normalization_1/ReadVariableOpReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_1/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┬
>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Б
-ResNet/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_1/BiasAdd:output:03ResNet/batch_normalization_1/ReadVariableOp:value:05ResNet/batch_normalization_1/ReadVariableOp_1:value:0DResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<║
+ResNet/batch_normalization_1/AssignNewValueAssignVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_1/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(─
-ResNet/batch_normalization_1/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_1/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Я
ResNet/activation_1/ReluRelu1ResNet/batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ш
#ResNet/conv2d/Conv2D/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
ResNet/conv2d/Conv2DConv2Dinputs+ResNet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
О
$ResNet/conv2d/BiasAdd/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╣
ResNet/conv2d/BiasAddBiasAddResNet/conv2d/Conv2D:output:0,ResNet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           Ь
%ResNet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ы
ResNet/conv2d_2/Conv2DConv2D&ResNet/activation_1/Relu:activations:0-ResNet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Т
&ResNet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
ResNet/conv2d_2/BiasAddBiasAddResNet/conv2d_2/Conv2D:output:0.ResNet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           Ш
)ResNet/batch_normalization/ReadVariableOpReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0Ь
+ResNet/batch_normalization/ReadVariableOp_1ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0║
:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ї
+ResNet/batch_normalization/FusedBatchNormV3FusedBatchNormV3ResNet/conv2d/BiasAdd:output:01ResNet/batch_normalization/ReadVariableOp:value:03ResNet/batch_normalization/ReadVariableOp_1:value:0BResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0DResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<▓
)ResNet/batch_normalization/AssignNewValueAssignVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource8ResNet/batch_normalization/FusedBatchNormV3:batch_mean:0;^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╝
+ResNet/batch_normalization/AssignNewValue_1AssignVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource<ResNet/batch_normalization/FusedBatchNormV3:batch_variance:0=^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ь
+ResNet/batch_normalization_2/ReadVariableOpReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_2/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┬
>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Б
-ResNet/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_2/BiasAdd:output:03ResNet/batch_normalization_2/ReadVariableOp:value:05ResNet/batch_normalization_2/ReadVariableOp_1:value:0DResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<║
+ResNet/batch_normalization_2/AssignNewValueAssignVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_2/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(─
-ResNet/batch_normalization_2/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_2/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Я
ResNet/activation_2/ReluRelu1ResNet/batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ы
ResNet/activation/ReluRelu/ResNet/batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           ▒
ResNet/add/addAddV2&ResNet/activation_2/Relu:activations:0$ResNet/activation/Relu:activations:0*
T0*A
_output_shapes/
-:+                           Ь
+ResNet/batch_normalization_3/ReadVariableOpReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_3/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┬
>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0є
-ResNet/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3ResNet/add/add:z:03ResNet/batch_normalization_3/ReadVariableOp:value:05ResNet/batch_normalization_3/ReadVariableOp_1:value:0DResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<║
+ResNet/batch_normalization_3/AssignNewValueAssignVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_3/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(─
-ResNet/batch_normalization_3/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_3/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Я
ResNet/activation_3/ReluRelu1ResNet/batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ь
%ResNet/conv2d_4/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ы
ResNet/conv2d_4/Conv2DConv2D&ResNet/activation_3/Relu:activations:0-ResNet/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Т
&ResNet/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
ResNet/conv2d_4/BiasAddBiasAddResNet/conv2d_4/Conv2D:output:0.ResNet/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            Ь
+ResNet/batch_normalization_5/ReadVariableOpReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_5/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Б
-ResNet/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_4/BiasAdd:output:03ResNet/batch_normalization_5/ReadVariableOp:value:05ResNet/batch_normalization_5/ReadVariableOp_1:value:0DResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<║
+ResNet/batch_normalization_5/AssignNewValueAssignVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_5/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(─
-ResNet/batch_normalization_5/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_5/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Я
ResNet/activation_5/ReluRelu1ResNet/batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            Ь
%ResNet/conv2d_3/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ы
ResNet/conv2d_3/Conv2DConv2D&ResNet/activation_3/Relu:activations:0-ResNet/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Т
&ResNet/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
ResNet/conv2d_3/BiasAddBiasAddResNet/conv2d_3/Conv2D:output:0.ResNet/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            Ь
%ResNet/conv2d_5/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ы
ResNet/conv2d_5/Conv2DConv2D&ResNet/activation_5/Relu:activations:0-ResNet/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Т
&ResNet/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
ResNet/conv2d_5/BiasAddBiasAddResNet/conv2d_5/Conv2D:output:0.ResNet/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            Ь
+ResNet/batch_normalization_4/ReadVariableOpReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_4/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Б
-ResNet/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_3/BiasAdd:output:03ResNet/batch_normalization_4/ReadVariableOp:value:05ResNet/batch_normalization_4/ReadVariableOp_1:value:0DResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<║
+ResNet/batch_normalization_4/AssignNewValueAssignVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_4/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(─
-ResNet/batch_normalization_4/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_4/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ь
+ResNet/batch_normalization_6/ReadVariableOpReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_6/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Б
-ResNet/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_5/BiasAdd:output:03ResNet/batch_normalization_6/ReadVariableOp:value:05ResNet/batch_normalization_6/ReadVariableOp_1:value:0DResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<║
+ResNet/batch_normalization_6/AssignNewValueAssignVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_6/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(─
-ResNet/batch_normalization_6/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_6/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Я
ResNet/activation_6/ReluRelu1ResNet/batch_normalization_6/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            Я
ResNet/activation_4/ReluRelu1ResNet/batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            ╡
ResNet/add_1/addAddV2&ResNet/activation_6/Relu:activations:0&ResNet/activation_4/Relu:activations:0*
T0*A
_output_shapes/
-:+                            Ь
+ResNet/batch_normalization_7/ReadVariableOpReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_7/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ї
-ResNet/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3ResNet/add_1/add:z:03ResNet/batch_normalization_7/ReadVariableOp:value:05ResNet/batch_normalization_7/ReadVariableOp_1:value:0DResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<║
+ResNet/batch_normalization_7/AssignNewValueAssignVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_7/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(─
-ResNet/batch_normalization_7/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_7/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Я
ResNet/activation_7/ReluRelu1ResNet/batch_normalization_7/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            Ь
%ResNet/conv2d_6/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0ы
ResNet/conv2d_6/Conv2DConv2D&ResNet/activation_7/Relu:activations:0-ResNet/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
*
paddingSAME*
strides
Т
&ResNet/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0┐
ResNet/conv2d_6/BiasAddBiasAddResNet/conv2d_6/Conv2D:output:0.ResNet/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
Р
ResNet/conv2d_6/SoftmaxSoftmax ResNet/conv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           
~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╠
 tf.compat.v1.transpose/transpose	Transpose!ResNet/conv2d_6/Softmax:softmax:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+
                           Z
$random_affine_transform_params/ShapeShapeinputs*
T0*
_output_shapes
:|
2random_affine_transform_params/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4random_affine_transform_params/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4random_affine_transform_params/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ь
,random_affine_transform_params/strided_sliceStridedSlice-random_affine_transform_params/Shape:output:0;random_affine_transform_params/strided_slice/stack:output:0=random_affine_transform_params/strided_slice/stack_1:output:0=random_affine_transform_params/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskа
3random_affine_transform_params/random_uniform/shapePack5random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:┼
;random_affine_transform_params/random_uniform/RandomUniformRandomUniform<random_affine_transform_params/random_uniform/shape:output:0*
T0*#
_output_shapes
:         *
dtype0б
$random_affine_transform_params/RoundRoundDrandom_affine_transform_params/random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:         i
$random_affine_transform_params/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @░
"random_affine_transform_params/mulMul(random_affine_transform_params/Round:y:0-random_affine_transform_params/mul/y:output:0*
T0*#
_output_shapes
:         i
$random_affine_transform_params/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?о
"random_affine_transform_params/subSub&random_affine_transform_params/mul:z:0-random_affine_transform_params/sub/y:output:0*
T0*#
_output_shapes
:         в
5random_affine_transform_params/random_uniform_1/shapePack5random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:x
3random_affine_transform_params/random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *█I└x
3random_affine_transform_params/random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *█I@╔
=random_affine_transform_params/random_uniform_1/RandomUniformRandomUniform>random_affine_transform_params/random_uniform_1/shape:output:0*
T0*#
_output_shapes
:         *
dtype0╫
3random_affine_transform_params/random_uniform_1/subSub<random_affine_transform_params/random_uniform_1/max:output:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*
_output_shapes
: щ
3random_affine_transform_params/random_uniform_1/mulMulFrandom_affine_transform_params/random_uniform_1/RandomUniform:output:07random_affine_transform_params/random_uniform_1/sub:z:0*
T0*#
_output_shapes
:         ▌
/random_affine_transform_params/random_uniform_1AddV27random_affine_transform_params/random_uniform_1/mul:z:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*#
_output_shapes
:         М
"random_affine_transform_params/CosCos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         М
"random_affine_transform_params/SinSin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         
"random_affine_transform_params/NegNeg&random_affine_transform_params/Sin:y:0*
T0*#
_output_shapes
:         й
$random_affine_transform_params/mul_1Mul&random_affine_transform_params/Neg:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:         О
$random_affine_transform_params/Sin_1Sin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         О
$random_affine_transform_params/Cos_1Cos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         л
$random_affine_transform_params/mul_2Mul(random_affine_transform_params/Cos_1:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:         ╝
'random_affine_transform_params/packed/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0*
N*
T0*'
_output_shapes
:         ╛
'random_affine_transform_params/packed/1Pack(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:         ╨
%random_affine_transform_params/packedPack0random_affine_transform_params/packed/0:output:00random_affine_transform_params/packed/1:output:0*
N*
T0*+
_output_shapes
:         В
-random_affine_transform_params/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╙
(random_affine_transform_params/transpose	Transpose.random_affine_transform_params/packed:output:06random_affine_transform_params/transpose/perm:output:0*
T0*+
_output_shapes
:         ╛
)random_affine_transform_params/packed_1/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0*
N*
T0*'
_output_shapes
:         └
)random_affine_transform_params/packed_1/1Pack(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:         ╓
'random_affine_transform_params/packed_1Pack2random_affine_transform_params/packed_1/0:output:02random_affine_transform_params/packed_1/1:output:0*
N*
T0*+
_output_shapes
:         Д
/random_affine_transform_params/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┘
*random_affine_transform_params/transpose_1	Transpose0random_affine_transform_params/packed_1:output:08random_affine_transform_params/transpose_1/perm:output:0*
T0*+
_output_shapes
:         }
$random_affine_transform_params/ConstConst*
_output_shapes

:*
dtype0*!
valueB"є5Cє5C
&random_affine_transform_params/Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   C╦
%random_affine_transform_params/MatMulBatchMatMulV2,random_affine_transform_params/transpose:y:0/random_affine_transform_params/Const_1:output:0*
T0*+
_output_shapes
:         └
$random_affine_transform_params/sub_1Sub-random_affine_transform_params/Const:output:0.random_affine_transform_params/MatMul:output:0*
T0*+
_output_shapes
:         ╚
'random_affine_transform_params/MatMul_1BatchMatMulV2.random_affine_transform_params/transpose_1:y:0(random_affine_transform_params/sub_1:z:0*
T0*+
_output_shapes
:         Й
4random_affine_transform_params/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Л
6random_affine_transform_params/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          Л
6random_affine_transform_params/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
.random_affine_transform_params/strided_slice_1StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_1/stack:output:0?random_affine_transform_params/strided_slice_1/stack_1:output:0?random_affine_transform_params/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_1Neg7random_affine_transform_params/strided_slice_1:output:0*
T0*#
_output_shapes
:         Й
4random_affine_transform_params/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Л
6random_affine_transform_params/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          Л
6random_affine_transform_params/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
.random_affine_transform_params/strided_slice_2StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_2/stack:output:0?random_affine_transform_params/strided_slice_2/stack_1:output:0?random_affine_transform_params/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_2Neg7random_affine_transform_params/strided_slice_2:output:0*
T0*#
_output_shapes
:         Г
$random_affine_transform_params/Neg_3Neg(random_affine_transform_params/Neg_1:y:0*
T0*#
_output_shapes
:         л
$random_affine_transform_params/mul_3Mul(random_affine_transform_params/Neg_3:y:0&random_affine_transform_params/Cos:y:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/mul_4Mul(random_affine_transform_params/Neg_2:y:0(random_affine_transform_params/mul_1:z:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/sub_2Sub(random_affine_transform_params/mul_3:z:0(random_affine_transform_params/mul_4:z:0*
T0*#
_output_shapes
:         Г
$random_affine_transform_params/Neg_4Neg(random_affine_transform_params/Neg_2:y:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/mul_5Mul(random_affine_transform_params/Neg_4:y:0(random_affine_transform_params/mul_2:z:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/mul_6Mul(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/Sin_1:y:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/sub_3Sub(random_affine_transform_params/mul_5:z:0(random_affine_transform_params/mul_6:z:0*
T0*#
_output_shapes
:         Е
2random_affine_transform_params/zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╨
,random_affine_transform_params/zeros/ReshapeReshape5random_affine_transform_params/strided_slice:output:0;random_affine_transform_params/zeros/Reshape/shape:output:0*
T0*
_output_shapes
:o
*random_affine_transform_params/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╞
$random_affine_transform_params/zerosFill5random_affine_transform_params/zeros/Reshape:output:03random_affine_transform_params/zeros/Const:output:0*
T0*#
_output_shapes
:         З
4random_affine_transform_params/zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╘
.random_affine_transform_params/zeros_1/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╠
&random_affine_transform_params/zeros_1Fill7random_affine_transform_params/zeros_1/Reshape:output:05random_affine_transform_params/zeros_1/Const:output:0*
T0*#
_output_shapes
:         ═
$random_affine_transform_params/stackPack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/sub_2:z:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/sub_3:z:0-random_affine_transform_params/zeros:output:0/random_affine_transform_params/zeros_1:output:0*
N*
T0*'
_output_shapes
:         *

axisЗ
4random_affine_transform_params/zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╘
.random_affine_transform_params/zeros_2/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╠
&random_affine_transform_params/zeros_2Fill7random_affine_transform_params/zeros_2/Reshape:output:05random_affine_transform_params/zeros_2/Const:output:0*
T0*#
_output_shapes
:         З
4random_affine_transform_params/zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╘
.random_affine_transform_params/zeros_3/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╠
&random_affine_transform_params/zeros_3Fill7random_affine_transform_params/zeros_3/Reshape:output:05random_affine_transform_params/zeros_3/Const:output:0*
T0*#
_output_shapes
:         ╤
&random_affine_transform_params/stack_1Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/Neg_2:y:0/random_affine_transform_params/zeros_2:output:0/random_affine_transform_params/zeros_3:output:0*
N*
T0*'
_output_shapes
:         *

axisЩ
Himage_projective_transform_layer/ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"v  v  Л
Fimage_projective_transform_layer/ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    У
;image_projective_transform_layer/ImageProjectiveTransformV3ImageProjectiveTransformV3inputs/random_affine_transform_params/stack_1:output:0Qimage_projective_transform_layer/ImageProjectiveTransformV3/output_shape:output:0Oimage_projective_transform_layer/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         ЎЎ*
dtype0*
interpolation
BILINEARО
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ╡
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+
                           Ю
'ResNet/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
ResNet/conv2d_1/Conv2D_1Conv2DPimage_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:0/ResNet/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
Ф
(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
ResNet/conv2d_1/BiasAdd_1BiasAdd!ResNet/conv2d_1/Conv2D_1:output:00ResNet/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎЮ
-ResNet/batch_normalization_1/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_1/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0ю
>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_1/AssignNewValue*
_output_shapes
:*
dtype0Ї
@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_1/AssignNewValue_1*
_output_shapes
:*
dtype0√
/ResNet/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_1/BiasAdd_1:output:05ResNet/batch_normalization_1/ReadVariableOp_2:value:05ResNet/batch_normalization_1/ReadVariableOp_3:value:0FResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<ю
-ResNet/batch_normalization_1/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_1/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_1/AssignNewValue?^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
-ResNet/batch_normalization_1/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_1/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_1/AssignNewValue_1A^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(У
ResNet/activation_1/Relu_1Relu3ResNet/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎЪ
%ResNet/conv2d/Conv2D_1/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Е
ResNet/conv2d/Conv2D_1Conv2DPimage_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:0-ResNet/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
Р
&ResNet/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
ResNet/conv2d/BiasAdd_1BiasAddResNet/conv2d/Conv2D_1:output:0.ResNet/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎЮ
'ResNet/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0с
ResNet/conv2d_2/Conv2D_1Conv2D(ResNet/activation_1/Relu_1:activations:0/ResNet/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
Ф
(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
ResNet/conv2d_2/BiasAdd_1BiasAdd!ResNet/conv2d_2/Conv2D_1:output:00ResNet/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎЪ
+ResNet/batch_normalization/ReadVariableOp_2ReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0Ь
+ResNet/batch_normalization/ReadVariableOp_3ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0ш
<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*^ResNet/batch_normalization/AssignNewValue*
_output_shapes
:*
dtype0ю
>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource,^ResNet/batch_normalization/AssignNewValue_1*
_output_shapes
:*
dtype0я
-ResNet/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3 ResNet/conv2d/BiasAdd_1:output:03ResNet/batch_normalization/ReadVariableOp_2:value:03ResNet/batch_normalization/ReadVariableOp_3:value:0DResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0FResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<ф
+ResNet/batch_normalization/AssignNewValue_2AssignVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization/FusedBatchNormV3_1:batch_mean:0*^ResNet/batch_normalization/AssignNewValue=^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ю
+ResNet/batch_normalization/AssignNewValue_3AssignVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization/FusedBatchNormV3_1:batch_variance:0,^ResNet/batch_normalization/AssignNewValue_1?^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ю
-ResNet/batch_normalization_2/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_2/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0ю
>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_2/AssignNewValue*
_output_shapes
:*
dtype0Ї
@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_2/AssignNewValue_1*
_output_shapes
:*
dtype0√
/ResNet/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_2/BiasAdd_1:output:05ResNet/batch_normalization_2/ReadVariableOp_2:value:05ResNet/batch_normalization_2/ReadVariableOp_3:value:0FResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<ю
-ResNet/batch_normalization_2/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_2/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_2/AssignNewValue?^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
-ResNet/batch_normalization_2/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_2/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_2/AssignNewValue_1A^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(У
ResNet/activation_2/Relu_1Relu3ResNet/batch_normalization_2/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎП
ResNet/activation/Relu_1Relu1ResNet/batch_normalization/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎз
ResNet/add/add_1AddV2(ResNet/activation_2/Relu_1:activations:0&ResNet/activation/Relu_1:activations:0*
T0*1
_output_shapes
:         ЎЎЮ
-ResNet/batch_normalization_3/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_3/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0ю
>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_3/AssignNewValue*
_output_shapes
:*
dtype0Ї
@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_3/AssignNewValue_1*
_output_shapes
:*
dtype0э
/ResNet/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3ResNet/add/add_1:z:05ResNet/batch_normalization_3/ReadVariableOp_2:value:05ResNet/batch_normalization_3/ReadVariableOp_3:value:0FResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<ю
-ResNet/batch_normalization_3/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_3/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_3/AssignNewValue?^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
-ResNet/batch_normalization_3/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_3/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_3/AssignNewValue_1A^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(У
ResNet/activation_3/Relu_1Relu3ResNet/batch_normalization_3/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎЮ
'ResNet/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0с
ResNet/conv2d_4/Conv2D_1Conv2D(ResNet/activation_3/Relu_1:activations:0/ResNet/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
Ф
(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
ResNet/conv2d_4/BiasAdd_1BiasAdd!ResNet/conv2d_4/Conv2D_1:output:00ResNet/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ Ю
-ResNet/batch_normalization_5/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_5/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0ю
>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_5/AssignNewValue*
_output_shapes
: *
dtype0Ї
@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_5/AssignNewValue_1*
_output_shapes
: *
dtype0√
/ResNet/batch_normalization_5/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_4/BiasAdd_1:output:05ResNet/batch_normalization_5/ReadVariableOp_2:value:05ResNet/batch_normalization_5/ReadVariableOp_3:value:0FResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<ю
-ResNet/batch_normalization_5/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_5/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_5/AssignNewValue?^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
-ResNet/batch_normalization_5/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_5/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_5/AssignNewValue_1A^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(У
ResNet/activation_5/Relu_1Relu3ResNet/batch_normalization_5/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ Ю
'ResNet/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0с
ResNet/conv2d_3/Conv2D_1Conv2D(ResNet/activation_3/Relu_1:activations:0/ResNet/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
Ф
(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
ResNet/conv2d_3/BiasAdd_1BiasAdd!ResNet/conv2d_3/Conv2D_1:output:00ResNet/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ Ю
'ResNet/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0с
ResNet/conv2d_5/Conv2D_1Conv2D(ResNet/activation_5/Relu_1:activations:0/ResNet/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
Ф
(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
ResNet/conv2d_5/BiasAdd_1BiasAdd!ResNet/conv2d_5/Conv2D_1:output:00ResNet/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ Ю
-ResNet/batch_normalization_4/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_4/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0ю
>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_4/AssignNewValue*
_output_shapes
: *
dtype0Ї
@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_4/AssignNewValue_1*
_output_shapes
: *
dtype0√
/ResNet/batch_normalization_4/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_3/BiasAdd_1:output:05ResNet/batch_normalization_4/ReadVariableOp_2:value:05ResNet/batch_normalization_4/ReadVariableOp_3:value:0FResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<ю
-ResNet/batch_normalization_4/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_4/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_4/AssignNewValue?^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
-ResNet/batch_normalization_4/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_4/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_4/AssignNewValue_1A^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ю
-ResNet/batch_normalization_6/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_6/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0ю
>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_6/AssignNewValue*
_output_shapes
: *
dtype0Ї
@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_6/AssignNewValue_1*
_output_shapes
: *
dtype0√
/ResNet/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_5/BiasAdd_1:output:05ResNet/batch_normalization_6/ReadVariableOp_2:value:05ResNet/batch_normalization_6/ReadVariableOp_3:value:0FResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<ю
-ResNet/batch_normalization_6/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_6/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_6/AssignNewValue?^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
-ResNet/batch_normalization_6/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_6/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_6/AssignNewValue_1A^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(У
ResNet/activation_6/Relu_1Relu3ResNet/batch_normalization_6/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ У
ResNet/activation_4/Relu_1Relu3ResNet/batch_normalization_4/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ л
ResNet/add_1/add_1AddV2(ResNet/activation_6/Relu_1:activations:0(ResNet/activation_4/Relu_1:activations:0*
T0*1
_output_shapes
:         ЎЎ Ю
-ResNet/batch_normalization_7/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_7/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0ю
>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_7/AssignNewValue*
_output_shapes
: *
dtype0Ї
@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_7/AssignNewValue_1*
_output_shapes
: *
dtype0я
/ResNet/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3ResNet/add_1/add_1:z:05ResNet/batch_normalization_7/ReadVariableOp_2:value:05ResNet/batch_normalization_7/ReadVariableOp_3:value:0FResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<ю
-ResNet/batch_normalization_7/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_7/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_7/AssignNewValue?^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(°
-ResNet/batch_normalization_7/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_7/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_7/AssignNewValue_1A^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(У
ResNet/activation_7/Relu_1Relu3ResNet/batch_normalization_7/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ Ю
'ResNet/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0с
ResNet/conv2d_6/Conv2D_1Conv2D(ResNet/activation_7/Relu_1:activations:0/ResNet/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ
*
paddingSAME*
strides
Ф
(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╡
ResNet/conv2d_6/BiasAdd_1BiasAdd!ResNet/conv2d_6/Conv2D_1:output:00ResNet/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ
Д
ResNet/conv2d_6/Softmax_1Softmax"ResNet/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:         ЎЎ
Ы
Jimage_projective_transform_layer_1/ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Н
Himage_projective_transform_layer_1/ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
=image_projective_transform_layer_1/ImageProjectiveTransformV3ImageProjectiveTransformV3#ResNet/conv2d_6/Softmax_1:softmax:0-random_affine_transform_params/stack:output:0Simage_projective_transform_layer_1/ImageProjectiveTransformV3/output_shape:output:0Qimage_projective_transform_layer_1/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         АА
*
dtype0*
interpolation
BILINEARА
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ё
"tf.compat.v1.transpose_1/transpose	TransposeRimage_projective_transform_layer_1/ImageProjectiveTransformV3:transformed_images:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА         
┌
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"
                  
*
paddingVALID*
strides
Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ў
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:

^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АAЦ
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:

x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       й
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:

У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:

`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:

s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         й
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        н
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:

С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:

^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:

z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:

{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"    ■   С
tf.math.reduce_sum_2/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
tf.math.reduce_mean/RankConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :│
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: К
IdentityIdentity!ResNet/conv2d_6/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+                           
a

Identity_1Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: ░1
NoOpNoOp*^ResNet/batch_normalization/AssignNewValue,^ResNet/batch_normalization/AssignNewValue_1,^ResNet/batch_normalization/AssignNewValue_2,^ResNet/batch_normalization/AssignNewValue_3;^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp=^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*^ResNet/batch_normalization/ReadVariableOp,^ResNet/batch_normalization/ReadVariableOp_1,^ResNet/batch_normalization/ReadVariableOp_2,^ResNet/batch_normalization/ReadVariableOp_3,^ResNet/batch_normalization_1/AssignNewValue.^ResNet/batch_normalization_1/AssignNewValue_1.^ResNet/batch_normalization_1/AssignNewValue_2.^ResNet/batch_normalization_1/AssignNewValue_3=^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_1/ReadVariableOp.^ResNet/batch_normalization_1/ReadVariableOp_1.^ResNet/batch_normalization_1/ReadVariableOp_2.^ResNet/batch_normalization_1/ReadVariableOp_3,^ResNet/batch_normalization_2/AssignNewValue.^ResNet/batch_normalization_2/AssignNewValue_1.^ResNet/batch_normalization_2/AssignNewValue_2.^ResNet/batch_normalization_2/AssignNewValue_3=^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_2/ReadVariableOp.^ResNet/batch_normalization_2/ReadVariableOp_1.^ResNet/batch_normalization_2/ReadVariableOp_2.^ResNet/batch_normalization_2/ReadVariableOp_3,^ResNet/batch_normalization_3/AssignNewValue.^ResNet/batch_normalization_3/AssignNewValue_1.^ResNet/batch_normalization_3/AssignNewValue_2.^ResNet/batch_normalization_3/AssignNewValue_3=^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_3/ReadVariableOp.^ResNet/batch_normalization_3/ReadVariableOp_1.^ResNet/batch_normalization_3/ReadVariableOp_2.^ResNet/batch_normalization_3/ReadVariableOp_3,^ResNet/batch_normalization_4/AssignNewValue.^ResNet/batch_normalization_4/AssignNewValue_1.^ResNet/batch_normalization_4/AssignNewValue_2.^ResNet/batch_normalization_4/AssignNewValue_3=^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_4/ReadVariableOp.^ResNet/batch_normalization_4/ReadVariableOp_1.^ResNet/batch_normalization_4/ReadVariableOp_2.^ResNet/batch_normalization_4/ReadVariableOp_3,^ResNet/batch_normalization_5/AssignNewValue.^ResNet/batch_normalization_5/AssignNewValue_1.^ResNet/batch_normalization_5/AssignNewValue_2.^ResNet/batch_normalization_5/AssignNewValue_3=^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_5/ReadVariableOp.^ResNet/batch_normalization_5/ReadVariableOp_1.^ResNet/batch_normalization_5/ReadVariableOp_2.^ResNet/batch_normalization_5/ReadVariableOp_3,^ResNet/batch_normalization_6/AssignNewValue.^ResNet/batch_normalization_6/AssignNewValue_1.^ResNet/batch_normalization_6/AssignNewValue_2.^ResNet/batch_normalization_6/AssignNewValue_3=^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_6/ReadVariableOp.^ResNet/batch_normalization_6/ReadVariableOp_1.^ResNet/batch_normalization_6/ReadVariableOp_2.^ResNet/batch_normalization_6/ReadVariableOp_3,^ResNet/batch_normalization_7/AssignNewValue.^ResNet/batch_normalization_7/AssignNewValue_1.^ResNet/batch_normalization_7/AssignNewValue_2.^ResNet/batch_normalization_7/AssignNewValue_3=^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_7/ReadVariableOp.^ResNet/batch_normalization_7/ReadVariableOp_1.^ResNet/batch_normalization_7/ReadVariableOp_2.^ResNet/batch_normalization_7/ReadVariableOp_3%^ResNet/conv2d/BiasAdd/ReadVariableOp'^ResNet/conv2d/BiasAdd_1/ReadVariableOp$^ResNet/conv2d/Conv2D/ReadVariableOp&^ResNet/conv2d/Conv2D_1/ReadVariableOp'^ResNet/conv2d_1/BiasAdd/ReadVariableOp)^ResNet/conv2d_1/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_1/Conv2D/ReadVariableOp(^ResNet/conv2d_1/Conv2D_1/ReadVariableOp'^ResNet/conv2d_2/BiasAdd/ReadVariableOp)^ResNet/conv2d_2/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_2/Conv2D/ReadVariableOp(^ResNet/conv2d_2/Conv2D_1/ReadVariableOp'^ResNet/conv2d_3/BiasAdd/ReadVariableOp)^ResNet/conv2d_3/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_3/Conv2D/ReadVariableOp(^ResNet/conv2d_3/Conv2D_1/ReadVariableOp'^ResNet/conv2d_4/BiasAdd/ReadVariableOp)^ResNet/conv2d_4/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_4/Conv2D/ReadVariableOp(^ResNet/conv2d_4/Conv2D_1/ReadVariableOp'^ResNet/conv2d_5/BiasAdd/ReadVariableOp)^ResNet/conv2d_5/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_5/Conv2D/ReadVariableOp(^ResNet/conv2d_5/Conv2D_1/ReadVariableOp'^ResNet/conv2d_6/BiasAdd/ReadVariableOp)^ResNet/conv2d_6/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_6/Conv2D/ReadVariableOp(^ResNet/conv2d_6/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)ResNet/batch_normalization/AssignNewValue)ResNet/batch_normalization/AssignNewValue2Z
+ResNet/batch_normalization/AssignNewValue_1+ResNet/batch_normalization/AssignNewValue_12Z
+ResNet/batch_normalization/AssignNewValue_2+ResNet/batch_normalization/AssignNewValue_22Z
+ResNet/batch_normalization/AssignNewValue_3+ResNet/batch_normalization/AssignNewValue_32x
:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp2|
<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_12|
<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2А
>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12V
)ResNet/batch_normalization/ReadVariableOp)ResNet/batch_normalization/ReadVariableOp2Z
+ResNet/batch_normalization/ReadVariableOp_1+ResNet/batch_normalization/ReadVariableOp_12Z
+ResNet/batch_normalization/ReadVariableOp_2+ResNet/batch_normalization/ReadVariableOp_22Z
+ResNet/batch_normalization/ReadVariableOp_3+ResNet/batch_normalization/ReadVariableOp_32Z
+ResNet/batch_normalization_1/AssignNewValue+ResNet/batch_normalization_1/AssignNewValue2^
-ResNet/batch_normalization_1/AssignNewValue_1-ResNet/batch_normalization_1/AssignNewValue_12^
-ResNet/batch_normalization_1/AssignNewValue_2-ResNet/batch_normalization_1/AssignNewValue_22^
-ResNet/batch_normalization_1/AssignNewValue_3-ResNet/batch_normalization_1/AssignNewValue_32|
<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_1/ReadVariableOp+ResNet/batch_normalization_1/ReadVariableOp2^
-ResNet/batch_normalization_1/ReadVariableOp_1-ResNet/batch_normalization_1/ReadVariableOp_12^
-ResNet/batch_normalization_1/ReadVariableOp_2-ResNet/batch_normalization_1/ReadVariableOp_22^
-ResNet/batch_normalization_1/ReadVariableOp_3-ResNet/batch_normalization_1/ReadVariableOp_32Z
+ResNet/batch_normalization_2/AssignNewValue+ResNet/batch_normalization_2/AssignNewValue2^
-ResNet/batch_normalization_2/AssignNewValue_1-ResNet/batch_normalization_2/AssignNewValue_12^
-ResNet/batch_normalization_2/AssignNewValue_2-ResNet/batch_normalization_2/AssignNewValue_22^
-ResNet/batch_normalization_2/AssignNewValue_3-ResNet/batch_normalization_2/AssignNewValue_32|
<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_2/ReadVariableOp+ResNet/batch_normalization_2/ReadVariableOp2^
-ResNet/batch_normalization_2/ReadVariableOp_1-ResNet/batch_normalization_2/ReadVariableOp_12^
-ResNet/batch_normalization_2/ReadVariableOp_2-ResNet/batch_normalization_2/ReadVariableOp_22^
-ResNet/batch_normalization_2/ReadVariableOp_3-ResNet/batch_normalization_2/ReadVariableOp_32Z
+ResNet/batch_normalization_3/AssignNewValue+ResNet/batch_normalization_3/AssignNewValue2^
-ResNet/batch_normalization_3/AssignNewValue_1-ResNet/batch_normalization_3/AssignNewValue_12^
-ResNet/batch_normalization_3/AssignNewValue_2-ResNet/batch_normalization_3/AssignNewValue_22^
-ResNet/batch_normalization_3/AssignNewValue_3-ResNet/batch_normalization_3/AssignNewValue_32|
<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_3/ReadVariableOp+ResNet/batch_normalization_3/ReadVariableOp2^
-ResNet/batch_normalization_3/ReadVariableOp_1-ResNet/batch_normalization_3/ReadVariableOp_12^
-ResNet/batch_normalization_3/ReadVariableOp_2-ResNet/batch_normalization_3/ReadVariableOp_22^
-ResNet/batch_normalization_3/ReadVariableOp_3-ResNet/batch_normalization_3/ReadVariableOp_32Z
+ResNet/batch_normalization_4/AssignNewValue+ResNet/batch_normalization_4/AssignNewValue2^
-ResNet/batch_normalization_4/AssignNewValue_1-ResNet/batch_normalization_4/AssignNewValue_12^
-ResNet/batch_normalization_4/AssignNewValue_2-ResNet/batch_normalization_4/AssignNewValue_22^
-ResNet/batch_normalization_4/AssignNewValue_3-ResNet/batch_normalization_4/AssignNewValue_32|
<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_4/ReadVariableOp+ResNet/batch_normalization_4/ReadVariableOp2^
-ResNet/batch_normalization_4/ReadVariableOp_1-ResNet/batch_normalization_4/ReadVariableOp_12^
-ResNet/batch_normalization_4/ReadVariableOp_2-ResNet/batch_normalization_4/ReadVariableOp_22^
-ResNet/batch_normalization_4/ReadVariableOp_3-ResNet/batch_normalization_4/ReadVariableOp_32Z
+ResNet/batch_normalization_5/AssignNewValue+ResNet/batch_normalization_5/AssignNewValue2^
-ResNet/batch_normalization_5/AssignNewValue_1-ResNet/batch_normalization_5/AssignNewValue_12^
-ResNet/batch_normalization_5/AssignNewValue_2-ResNet/batch_normalization_5/AssignNewValue_22^
-ResNet/batch_normalization_5/AssignNewValue_3-ResNet/batch_normalization_5/AssignNewValue_32|
<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_5/ReadVariableOp+ResNet/batch_normalization_5/ReadVariableOp2^
-ResNet/batch_normalization_5/ReadVariableOp_1-ResNet/batch_normalization_5/ReadVariableOp_12^
-ResNet/batch_normalization_5/ReadVariableOp_2-ResNet/batch_normalization_5/ReadVariableOp_22^
-ResNet/batch_normalization_5/ReadVariableOp_3-ResNet/batch_normalization_5/ReadVariableOp_32Z
+ResNet/batch_normalization_6/AssignNewValue+ResNet/batch_normalization_6/AssignNewValue2^
-ResNet/batch_normalization_6/AssignNewValue_1-ResNet/batch_normalization_6/AssignNewValue_12^
-ResNet/batch_normalization_6/AssignNewValue_2-ResNet/batch_normalization_6/AssignNewValue_22^
-ResNet/batch_normalization_6/AssignNewValue_3-ResNet/batch_normalization_6/AssignNewValue_32|
<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_6/ReadVariableOp+ResNet/batch_normalization_6/ReadVariableOp2^
-ResNet/batch_normalization_6/ReadVariableOp_1-ResNet/batch_normalization_6/ReadVariableOp_12^
-ResNet/batch_normalization_6/ReadVariableOp_2-ResNet/batch_normalization_6/ReadVariableOp_22^
-ResNet/batch_normalization_6/ReadVariableOp_3-ResNet/batch_normalization_6/ReadVariableOp_32Z
+ResNet/batch_normalization_7/AssignNewValue+ResNet/batch_normalization_7/AssignNewValue2^
-ResNet/batch_normalization_7/AssignNewValue_1-ResNet/batch_normalization_7/AssignNewValue_12^
-ResNet/batch_normalization_7/AssignNewValue_2-ResNet/batch_normalization_7/AssignNewValue_22^
-ResNet/batch_normalization_7/AssignNewValue_3-ResNet/batch_normalization_7/AssignNewValue_32|
<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_7/ReadVariableOp+ResNet/batch_normalization_7/ReadVariableOp2^
-ResNet/batch_normalization_7/ReadVariableOp_1-ResNet/batch_normalization_7/ReadVariableOp_12^
-ResNet/batch_normalization_7/ReadVariableOp_2-ResNet/batch_normalization_7/ReadVariableOp_22^
-ResNet/batch_normalization_7/ReadVariableOp_3-ResNet/batch_normalization_7/ReadVariableOp_32L
$ResNet/conv2d/BiasAdd/ReadVariableOp$ResNet/conv2d/BiasAdd/ReadVariableOp2P
&ResNet/conv2d/BiasAdd_1/ReadVariableOp&ResNet/conv2d/BiasAdd_1/ReadVariableOp2J
#ResNet/conv2d/Conv2D/ReadVariableOp#ResNet/conv2d/Conv2D/ReadVariableOp2N
%ResNet/conv2d/Conv2D_1/ReadVariableOp%ResNet/conv2d/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_1/BiasAdd/ReadVariableOp&ResNet/conv2d_1/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_1/BiasAdd_1/ReadVariableOp(ResNet/conv2d_1/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_1/Conv2D/ReadVariableOp%ResNet/conv2d_1/Conv2D/ReadVariableOp2R
'ResNet/conv2d_1/Conv2D_1/ReadVariableOp'ResNet/conv2d_1/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_2/BiasAdd/ReadVariableOp&ResNet/conv2d_2/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_2/BiasAdd_1/ReadVariableOp(ResNet/conv2d_2/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_2/Conv2D/ReadVariableOp%ResNet/conv2d_2/Conv2D/ReadVariableOp2R
'ResNet/conv2d_2/Conv2D_1/ReadVariableOp'ResNet/conv2d_2/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_3/BiasAdd/ReadVariableOp&ResNet/conv2d_3/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_3/BiasAdd_1/ReadVariableOp(ResNet/conv2d_3/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_3/Conv2D/ReadVariableOp%ResNet/conv2d_3/Conv2D/ReadVariableOp2R
'ResNet/conv2d_3/Conv2D_1/ReadVariableOp'ResNet/conv2d_3/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_4/BiasAdd/ReadVariableOp&ResNet/conv2d_4/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_4/BiasAdd_1/ReadVariableOp(ResNet/conv2d_4/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_4/Conv2D/ReadVariableOp%ResNet/conv2d_4/Conv2D/ReadVariableOp2R
'ResNet/conv2d_4/Conv2D_1/ReadVariableOp'ResNet/conv2d_4/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_5/BiasAdd/ReadVariableOp&ResNet/conv2d_5/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_5/BiasAdd_1/ReadVariableOp(ResNet/conv2d_5/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_5/Conv2D/ReadVariableOp%ResNet/conv2d_5/Conv2D/ReadVariableOp2R
'ResNet/conv2d_5/Conv2D_1/ReadVariableOp'ResNet/conv2d_5/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_6/BiasAdd/ReadVariableOp&ResNet/conv2d_6/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_6/BiasAdd_1/ReadVariableOp(ResNet/conv2d_6/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_6/Conv2D/ReadVariableOp%ResNet/conv2d_6/Conv2D/ReadVariableOp2R
'ResNet/conv2d_6/Conv2D_1/ReadVariableOp'ResNet/conv2d_6/Conv2D_1/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16803
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
З
┴
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386970

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш	
╥
7__inference_batch_normalization_7_layer_call_fn_1392253

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387323Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
█
n
B__inference_add_1_layer_call_and_return_conditional_losses_1392240
inputs_0
inputs_1
identityl
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+                            i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+                            :+                            :k g
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+                            
"
_user_specified_name
inputs_1
Т	
╨
5__inference_batch_normalization_layer_call_fn_1391815

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387034Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Б
№
C__inference_conv2d_layer_call_and_return_conditional_losses_1391727

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16883
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
л
J
"__inference__update_step_xla_16813
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ф
J
.__inference_activation_4_layer_call_fn_1392223

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1387586z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
з
╘

(__inference_ResNet_layer_call_fn_1391035

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╓
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1387630Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_4_layer_call_fn_1392172

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387290Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16863
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
╡
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1387402

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392284

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Г
■
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1387502

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Г
■
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1392065

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387323

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╡
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1387586

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392036

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ш▀
╓@
B__inference_model_layer_call_and_return_conditional_losses_1390524

inputsH
.resnet_conv2d_1_conv2d_readvariableop_resource:=
/resnet_conv2d_1_biasadd_readvariableop_resource:B
4resnet_batch_normalization_1_readvariableop_resource:D
6resnet_batch_normalization_1_readvariableop_1_resource:S
Eresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:U
Gresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:F
,resnet_conv2d_conv2d_readvariableop_resource:;
-resnet_conv2d_biasadd_readvariableop_resource:H
.resnet_conv2d_2_conv2d_readvariableop_resource:=
/resnet_conv2d_2_biasadd_readvariableop_resource:@
2resnet_batch_normalization_readvariableop_resource:B
4resnet_batch_normalization_readvariableop_1_resource:Q
Cresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:S
Eresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:B
4resnet_batch_normalization_2_readvariableop_resource:D
6resnet_batch_normalization_2_readvariableop_1_resource:S
Eresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:U
Gresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:B
4resnet_batch_normalization_3_readvariableop_resource:D
6resnet_batch_normalization_3_readvariableop_1_resource:S
Eresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:U
Gresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:H
.resnet_conv2d_4_conv2d_readvariableop_resource: =
/resnet_conv2d_4_biasadd_readvariableop_resource: B
4resnet_batch_normalization_5_readvariableop_resource: D
6resnet_batch_normalization_5_readvariableop_1_resource: S
Eresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: H
.resnet_conv2d_3_conv2d_readvariableop_resource: =
/resnet_conv2d_3_biasadd_readvariableop_resource: H
.resnet_conv2d_5_conv2d_readvariableop_resource:  =
/resnet_conv2d_5_biasadd_readvariableop_resource: B
4resnet_batch_normalization_4_readvariableop_resource: D
6resnet_batch_normalization_4_readvariableop_1_resource: S
Eresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: B
4resnet_batch_normalization_6_readvariableop_resource: D
6resnet_batch_normalization_6_readvariableop_1_resource: S
Eresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: B
4resnet_batch_normalization_7_readvariableop_resource: D
6resnet_batch_normalization_7_readvariableop_1_resource: S
Eresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource: U
Gresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: H
.resnet_conv2d_6_conv2d_readvariableop_resource: 
=
/resnet_conv2d_6_biasadd_readvariableop_resource:

identity

identity_1Ив:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpв<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1в<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpв>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1в)ResNet/batch_normalization/ReadVariableOpв+ResNet/batch_normalization/ReadVariableOp_1в+ResNet/batch_normalization/ReadVariableOp_2в+ResNet/batch_normalization/ReadVariableOp_3в<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_1/ReadVariableOpв-ResNet/batch_normalization_1/ReadVariableOp_1в-ResNet/batch_normalization_1/ReadVariableOp_2в-ResNet/batch_normalization_1/ReadVariableOp_3в<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_2/ReadVariableOpв-ResNet/batch_normalization_2/ReadVariableOp_1в-ResNet/batch_normalization_2/ReadVariableOp_2в-ResNet/batch_normalization_2/ReadVariableOp_3в<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_3/ReadVariableOpв-ResNet/batch_normalization_3/ReadVariableOp_1в-ResNet/batch_normalization_3/ReadVariableOp_2в-ResNet/batch_normalization_3/ReadVariableOp_3в<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_4/ReadVariableOpв-ResNet/batch_normalization_4/ReadVariableOp_1в-ResNet/batch_normalization_4/ReadVariableOp_2в-ResNet/batch_normalization_4/ReadVariableOp_3в<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_5/ReadVariableOpв-ResNet/batch_normalization_5/ReadVariableOp_1в-ResNet/batch_normalization_5/ReadVariableOp_2в-ResNet/batch_normalization_5/ReadVariableOp_3в<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_6/ReadVariableOpв-ResNet/batch_normalization_6/ReadVariableOp_1в-ResNet/batch_normalization_6/ReadVariableOp_2в-ResNet/batch_normalization_6/ReadVariableOp_3в<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpв>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpв@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1в+ResNet/batch_normalization_7/ReadVariableOpв-ResNet/batch_normalization_7/ReadVariableOp_1в-ResNet/batch_normalization_7/ReadVariableOp_2в-ResNet/batch_normalization_7/ReadVariableOp_3в$ResNet/conv2d/BiasAdd/ReadVariableOpв&ResNet/conv2d/BiasAdd_1/ReadVariableOpв#ResNet/conv2d/Conv2D/ReadVariableOpв%ResNet/conv2d/Conv2D_1/ReadVariableOpв&ResNet/conv2d_1/BiasAdd/ReadVariableOpв(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_1/Conv2D/ReadVariableOpв'ResNet/conv2d_1/Conv2D_1/ReadVariableOpв&ResNet/conv2d_2/BiasAdd/ReadVariableOpв(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_2/Conv2D/ReadVariableOpв'ResNet/conv2d_2/Conv2D_1/ReadVariableOpв&ResNet/conv2d_3/BiasAdd/ReadVariableOpв(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_3/Conv2D/ReadVariableOpв'ResNet/conv2d_3/Conv2D_1/ReadVariableOpв&ResNet/conv2d_4/BiasAdd/ReadVariableOpв(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_4/Conv2D/ReadVariableOpв'ResNet/conv2d_4/Conv2D_1/ReadVariableOpв&ResNet/conv2d_5/BiasAdd/ReadVariableOpв(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_5/Conv2D/ReadVariableOpв'ResNet/conv2d_5/Conv2D_1/ReadVariableOpв&ResNet/conv2d_6/BiasAdd/ReadVariableOpв(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpв%ResNet/conv2d_6/Conv2D/ReadVariableOpв'ResNet/conv2d_6/Conv2D_1/ReadVariableOpЬ
%ResNet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╦
ResNet/conv2d_1/Conv2DConv2Dinputs-ResNet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Т
&ResNet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
ResNet/conv2d_1/BiasAddBiasAddResNet/conv2d_1/Conv2D:output:0.ResNet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           Ь
+ResNet/batch_normalization_1/ReadVariableOpReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_1/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┬
>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0є
-ResNet/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_1/BiasAdd:output:03ResNet/batch_normalization_1/ReadVariableOp:value:05ResNet/batch_normalization_1/ReadVariableOp_1:value:0DResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( Я
ResNet/activation_1/ReluRelu1ResNet/batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ш
#ResNet/conv2d/Conv2D/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
ResNet/conv2d/Conv2DConv2Dinputs+ResNet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
О
$ResNet/conv2d/BiasAdd/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╣
ResNet/conv2d/BiasAddBiasAddResNet/conv2d/Conv2D:output:0,ResNet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           Ь
%ResNet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ы
ResNet/conv2d_2/Conv2DConv2D&ResNet/activation_1/Relu:activations:0-ResNet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Т
&ResNet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
ResNet/conv2d_2/BiasAddBiasAddResNet/conv2d_2/Conv2D:output:0.ResNet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           Ш
)ResNet/batch_normalization/ReadVariableOpReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0Ь
+ResNet/batch_normalization/ReadVariableOp_1ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0║
:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ч
+ResNet/batch_normalization/FusedBatchNormV3FusedBatchNormV3ResNet/conv2d/BiasAdd:output:01ResNet/batch_normalization/ReadVariableOp:value:03ResNet/batch_normalization/ReadVariableOp_1:value:0BResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0DResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( Ь
+ResNet/batch_normalization_2/ReadVariableOpReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_2/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┬
>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0є
-ResNet/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_2/BiasAdd:output:03ResNet/batch_normalization_2/ReadVariableOp:value:05ResNet/batch_normalization_2/ReadVariableOp_1:value:0DResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( Я
ResNet/activation_2/ReluRelu1ResNet/batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ы
ResNet/activation/ReluRelu/ResNet/batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           ▒
ResNet/add/addAddV2&ResNet/activation_2/Relu:activations:0$ResNet/activation/Relu:activations:0*
T0*A
_output_shapes/
-:+                           Ь
+ResNet/batch_normalization_3/ReadVariableOpReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_3/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0╛
<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┬
>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0х
-ResNet/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3ResNet/add/add:z:03ResNet/batch_normalization_3/ReadVariableOp:value:05ResNet/batch_normalization_3/ReadVariableOp_1:value:0DResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( Я
ResNet/activation_3/ReluRelu1ResNet/batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ь
%ResNet/conv2d_4/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ы
ResNet/conv2d_4/Conv2DConv2D&ResNet/activation_3/Relu:activations:0-ResNet/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Т
&ResNet/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
ResNet/conv2d_4/BiasAddBiasAddResNet/conv2d_4/Conv2D:output:0.ResNet/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            Ь
+ResNet/batch_normalization_5/ReadVariableOpReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_5/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0є
-ResNet/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_4/BiasAdd:output:03ResNet/batch_normalization_5/ReadVariableOp:value:05ResNet/batch_normalization_5/ReadVariableOp_1:value:0DResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( Я
ResNet/activation_5/ReluRelu1ResNet/batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            Ь
%ResNet/conv2d_3/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ы
ResNet/conv2d_3/Conv2DConv2D&ResNet/activation_3/Relu:activations:0-ResNet/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Т
&ResNet/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
ResNet/conv2d_3/BiasAddBiasAddResNet/conv2d_3/Conv2D:output:0.ResNet/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            Ь
%ResNet/conv2d_5/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ы
ResNet/conv2d_5/Conv2DConv2D&ResNet/activation_5/Relu:activations:0-ResNet/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Т
&ResNet/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
ResNet/conv2d_5/BiasAddBiasAddResNet/conv2d_5/Conv2D:output:0.ResNet/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            Ь
+ResNet/batch_normalization_4/ReadVariableOpReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_4/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0є
-ResNet/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_3/BiasAdd:output:03ResNet/batch_normalization_4/ReadVariableOp:value:05ResNet/batch_normalization_4/ReadVariableOp_1:value:0DResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( Ь
+ResNet/batch_normalization_6/ReadVariableOpReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_6/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0є
-ResNet/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_5/BiasAdd:output:03ResNet/batch_normalization_6/ReadVariableOp:value:05ResNet/batch_normalization_6/ReadVariableOp_1:value:0DResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( Я
ResNet/activation_6/ReluRelu1ResNet/batch_normalization_6/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            Я
ResNet/activation_4/ReluRelu1ResNet/batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            ╡
ResNet/add_1/addAddV2&ResNet/activation_6/Relu:activations:0&ResNet/activation_4/Relu:activations:0*
T0*A
_output_shapes/
-:+                            Ь
+ResNet/batch_normalization_7/ReadVariableOpReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_7/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0╛
<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┬
>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ч
-ResNet/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3ResNet/add_1/add:z:03ResNet/batch_normalization_7/ReadVariableOp:value:05ResNet/batch_normalization_7/ReadVariableOp_1:value:0DResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( Я
ResNet/activation_7/ReluRelu1ResNet/batch_normalization_7/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            Ь
%ResNet/conv2d_6/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0ы
ResNet/conv2d_6/Conv2DConv2D&ResNet/activation_7/Relu:activations:0-ResNet/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
*
paddingSAME*
strides
Т
&ResNet/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0┐
ResNet/conv2d_6/BiasAddBiasAddResNet/conv2d_6/Conv2D:output:0.ResNet/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
Р
ResNet/conv2d_6/SoftmaxSoftmax ResNet/conv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           
~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╠
 tf.compat.v1.transpose/transpose	Transpose!ResNet/conv2d_6/Softmax:softmax:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+
                           Z
$random_affine_transform_params/ShapeShapeinputs*
T0*
_output_shapes
:|
2random_affine_transform_params/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4random_affine_transform_params/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4random_affine_transform_params/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ь
,random_affine_transform_params/strided_sliceStridedSlice-random_affine_transform_params/Shape:output:0;random_affine_transform_params/strided_slice/stack:output:0=random_affine_transform_params/strided_slice/stack_1:output:0=random_affine_transform_params/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskа
3random_affine_transform_params/random_uniform/shapePack5random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:┼
;random_affine_transform_params/random_uniform/RandomUniformRandomUniform<random_affine_transform_params/random_uniform/shape:output:0*
T0*#
_output_shapes
:         *
dtype0б
$random_affine_transform_params/RoundRoundDrandom_affine_transform_params/random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:         i
$random_affine_transform_params/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @░
"random_affine_transform_params/mulMul(random_affine_transform_params/Round:y:0-random_affine_transform_params/mul/y:output:0*
T0*#
_output_shapes
:         i
$random_affine_transform_params/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?о
"random_affine_transform_params/subSub&random_affine_transform_params/mul:z:0-random_affine_transform_params/sub/y:output:0*
T0*#
_output_shapes
:         в
5random_affine_transform_params/random_uniform_1/shapePack5random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:x
3random_affine_transform_params/random_uniform_1/minConst*
_output_shapes
: *
dtype0*
valueB
 *█I└x
3random_affine_transform_params/random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *█I@╔
=random_affine_transform_params/random_uniform_1/RandomUniformRandomUniform>random_affine_transform_params/random_uniform_1/shape:output:0*
T0*#
_output_shapes
:         *
dtype0╫
3random_affine_transform_params/random_uniform_1/subSub<random_affine_transform_params/random_uniform_1/max:output:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*
_output_shapes
: щ
3random_affine_transform_params/random_uniform_1/mulMulFrandom_affine_transform_params/random_uniform_1/RandomUniform:output:07random_affine_transform_params/random_uniform_1/sub:z:0*
T0*#
_output_shapes
:         ▌
/random_affine_transform_params/random_uniform_1AddV27random_affine_transform_params/random_uniform_1/mul:z:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*#
_output_shapes
:         М
"random_affine_transform_params/CosCos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         М
"random_affine_transform_params/SinSin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         
"random_affine_transform_params/NegNeg&random_affine_transform_params/Sin:y:0*
T0*#
_output_shapes
:         й
$random_affine_transform_params/mul_1Mul&random_affine_transform_params/Neg:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:         О
$random_affine_transform_params/Sin_1Sin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         О
$random_affine_transform_params/Cos_1Cos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:         л
$random_affine_transform_params/mul_2Mul(random_affine_transform_params/Cos_1:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:         ╝
'random_affine_transform_params/packed/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0*
N*
T0*'
_output_shapes
:         ╛
'random_affine_transform_params/packed/1Pack(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:         ╨
%random_affine_transform_params/packedPack0random_affine_transform_params/packed/0:output:00random_affine_transform_params/packed/1:output:0*
N*
T0*+
_output_shapes
:         В
-random_affine_transform_params/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╙
(random_affine_transform_params/transpose	Transpose.random_affine_transform_params/packed:output:06random_affine_transform_params/transpose/perm:output:0*
T0*+
_output_shapes
:         ╛
)random_affine_transform_params/packed_1/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0*
N*
T0*'
_output_shapes
:         └
)random_affine_transform_params/packed_1/1Pack(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:         ╓
'random_affine_transform_params/packed_1Pack2random_affine_transform_params/packed_1/0:output:02random_affine_transform_params/packed_1/1:output:0*
N*
T0*+
_output_shapes
:         Д
/random_affine_transform_params/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┘
*random_affine_transform_params/transpose_1	Transpose0random_affine_transform_params/packed_1:output:08random_affine_transform_params/transpose_1/perm:output:0*
T0*+
_output_shapes
:         }
$random_affine_transform_params/ConstConst*
_output_shapes

:*
dtype0*!
valueB"є5Cє5C
&random_affine_transform_params/Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   C╦
%random_affine_transform_params/MatMulBatchMatMulV2,random_affine_transform_params/transpose:y:0/random_affine_transform_params/Const_1:output:0*
T0*+
_output_shapes
:         └
$random_affine_transform_params/sub_1Sub-random_affine_transform_params/Const:output:0.random_affine_transform_params/MatMul:output:0*
T0*+
_output_shapes
:         ╚
'random_affine_transform_params/MatMul_1BatchMatMulV2.random_affine_transform_params/transpose_1:y:0(random_affine_transform_params/sub_1:z:0*
T0*+
_output_shapes
:         Й
4random_affine_transform_params/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Л
6random_affine_transform_params/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          Л
6random_affine_transform_params/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
.random_affine_transform_params/strided_slice_1StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_1/stack:output:0?random_affine_transform_params/strided_slice_1/stack_1:output:0?random_affine_transform_params/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_1Neg7random_affine_transform_params/strided_slice_1:output:0*
T0*#
_output_shapes
:         Й
4random_affine_transform_params/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Л
6random_affine_transform_params/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          Л
6random_affine_transform_params/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ж
.random_affine_transform_params/strided_slice_2StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_2/stack:output:0?random_affine_transform_params/strided_slice_2/stack_1:output:0?random_affine_transform_params/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_2Neg7random_affine_transform_params/strided_slice_2:output:0*
T0*#
_output_shapes
:         Г
$random_affine_transform_params/Neg_3Neg(random_affine_transform_params/Neg_1:y:0*
T0*#
_output_shapes
:         л
$random_affine_transform_params/mul_3Mul(random_affine_transform_params/Neg_3:y:0&random_affine_transform_params/Cos:y:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/mul_4Mul(random_affine_transform_params/Neg_2:y:0(random_affine_transform_params/mul_1:z:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/sub_2Sub(random_affine_transform_params/mul_3:z:0(random_affine_transform_params/mul_4:z:0*
T0*#
_output_shapes
:         Г
$random_affine_transform_params/Neg_4Neg(random_affine_transform_params/Neg_2:y:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/mul_5Mul(random_affine_transform_params/Neg_4:y:0(random_affine_transform_params/mul_2:z:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/mul_6Mul(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/Sin_1:y:0*
T0*#
_output_shapes
:         н
$random_affine_transform_params/sub_3Sub(random_affine_transform_params/mul_5:z:0(random_affine_transform_params/mul_6:z:0*
T0*#
_output_shapes
:         Е
2random_affine_transform_params/zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╨
,random_affine_transform_params/zeros/ReshapeReshape5random_affine_transform_params/strided_slice:output:0;random_affine_transform_params/zeros/Reshape/shape:output:0*
T0*
_output_shapes
:o
*random_affine_transform_params/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╞
$random_affine_transform_params/zerosFill5random_affine_transform_params/zeros/Reshape:output:03random_affine_transform_params/zeros/Const:output:0*
T0*#
_output_shapes
:         З
4random_affine_transform_params/zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╘
.random_affine_transform_params/zeros_1/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╠
&random_affine_transform_params/zeros_1Fill7random_affine_transform_params/zeros_1/Reshape:output:05random_affine_transform_params/zeros_1/Const:output:0*
T0*#
_output_shapes
:         ═
$random_affine_transform_params/stackPack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/sub_2:z:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/sub_3:z:0-random_affine_transform_params/zeros:output:0/random_affine_transform_params/zeros_1:output:0*
N*
T0*'
_output_shapes
:         *

axisЗ
4random_affine_transform_params/zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╘
.random_affine_transform_params/zeros_2/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╠
&random_affine_transform_params/zeros_2Fill7random_affine_transform_params/zeros_2/Reshape:output:05random_affine_transform_params/zeros_2/Const:output:0*
T0*#
_output_shapes
:         З
4random_affine_transform_params/zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╘
.random_affine_transform_params/zeros_3/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╠
&random_affine_transform_params/zeros_3Fill7random_affine_transform_params/zeros_3/Reshape:output:05random_affine_transform_params/zeros_3/Const:output:0*
T0*#
_output_shapes
:         ╤
&random_affine_transform_params/stack_1Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/Neg_2:y:0/random_affine_transform_params/zeros_2:output:0/random_affine_transform_params/zeros_3:output:0*
N*
T0*'
_output_shapes
:         *

axisЩ
Himage_projective_transform_layer/ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"v  v  Л
Fimage_projective_transform_layer/ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    У
;image_projective_transform_layer/ImageProjectiveTransformV3ImageProjectiveTransformV3inputs/random_affine_transform_params/stack_1:output:0Qimage_projective_transform_layer/ImageProjectiveTransformV3/output_shape:output:0Oimage_projective_transform_layer/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         ЎЎ*
dtype0*
interpolation
BILINEARО
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ╡
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+
                           Ю
'ResNet/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Й
ResNet/conv2d_1/Conv2D_1Conv2DPimage_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:0/ResNet/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
Ф
(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
ResNet/conv2d_1/BiasAdd_1BiasAdd!ResNet/conv2d_1/Conv2D_1:output:00ResNet/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎЮ
-ResNet/batch_normalization_1/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_1/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0└
>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0─
@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0э
/ResNet/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_1/BiasAdd_1:output:05ResNet/batch_normalization_1/ReadVariableOp_2:value:05ResNet/batch_normalization_1/ReadVariableOp_3:value:0FResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( У
ResNet/activation_1/Relu_1Relu3ResNet/batch_normalization_1/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎЪ
%ResNet/conv2d/Conv2D_1/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Е
ResNet/conv2d/Conv2D_1Conv2DPimage_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:0-ResNet/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
Р
&ResNet/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
ResNet/conv2d/BiasAdd_1BiasAddResNet/conv2d/Conv2D_1:output:0.ResNet/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎЮ
'ResNet/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0с
ResNet/conv2d_2/Conv2D_1Conv2D(ResNet/activation_1/Relu_1:activations:0/ResNet/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ*
paddingSAME*
strides
Ф
(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
ResNet/conv2d_2/BiasAdd_1BiasAdd!ResNet/conv2d_2/Conv2D_1:output:00ResNet/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎЪ
+ResNet/batch_normalization/ReadVariableOp_2ReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0Ь
+ResNet/batch_normalization/ReadVariableOp_3ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0╝
<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0└
>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0с
-ResNet/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3 ResNet/conv2d/BiasAdd_1:output:03ResNet/batch_normalization/ReadVariableOp_2:value:03ResNet/batch_normalization/ReadVariableOp_3:value:0DResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0FResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( Ю
-ResNet/batch_normalization_2/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_2/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0└
>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0─
@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0э
/ResNet/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_2/BiasAdd_1:output:05ResNet/batch_normalization_2/ReadVariableOp_2:value:05ResNet/batch_normalization_2/ReadVariableOp_3:value:0FResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( У
ResNet/activation_2/Relu_1Relu3ResNet/batch_normalization_2/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎП
ResNet/activation/Relu_1Relu1ResNet/batch_normalization/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎз
ResNet/add/add_1AddV2(ResNet/activation_2/Relu_1:activations:0&ResNet/activation/Relu_1:activations:0*
T0*1
_output_shapes
:         ЎЎЮ
-ResNet/batch_normalization_3/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0а
-ResNet/batch_normalization_3/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0└
>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0─
@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0▀
/ResNet/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3ResNet/add/add_1:z:05ResNet/batch_normalization_3/ReadVariableOp_2:value:05ResNet/batch_normalization_3/ReadVariableOp_3:value:0FResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ:::::*
epsilon%oГ:*
is_training( У
ResNet/activation_3/Relu_1Relu3ResNet/batch_normalization_3/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎЮ
'ResNet/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0с
ResNet/conv2d_4/Conv2D_1Conv2D(ResNet/activation_3/Relu_1:activations:0/ResNet/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
Ф
(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
ResNet/conv2d_4/BiasAdd_1BiasAdd!ResNet/conv2d_4/Conv2D_1:output:00ResNet/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ Ю
-ResNet/batch_normalization_5/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_5/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0└
>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0─
@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0э
/ResNet/batch_normalization_5/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_4/BiasAdd_1:output:05ResNet/batch_normalization_5/ReadVariableOp_2:value:05ResNet/batch_normalization_5/ReadVariableOp_3:value:0FResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( У
ResNet/activation_5/Relu_1Relu3ResNet/batch_normalization_5/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ Ю
'ResNet/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0с
ResNet/conv2d_3/Conv2D_1Conv2D(ResNet/activation_3/Relu_1:activations:0/ResNet/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
Ф
(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
ResNet/conv2d_3/BiasAdd_1BiasAdd!ResNet/conv2d_3/Conv2D_1:output:00ResNet/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ Ю
'ResNet/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0с
ResNet/conv2d_5/Conv2D_1Conv2D(ResNet/activation_5/Relu_1:activations:0/ResNet/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ *
paddingSAME*
strides
Ф
(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
ResNet/conv2d_5/BiasAdd_1BiasAdd!ResNet/conv2d_5/Conv2D_1:output:00ResNet/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ Ю
-ResNet/batch_normalization_4/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_4/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0└
>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0─
@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0э
/ResNet/batch_normalization_4/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_3/BiasAdd_1:output:05ResNet/batch_normalization_4/ReadVariableOp_2:value:05ResNet/batch_normalization_4/ReadVariableOp_3:value:0FResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( Ю
-ResNet/batch_normalization_6/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_6/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0└
>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0─
@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0э
/ResNet/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_5/BiasAdd_1:output:05ResNet/batch_normalization_6/ReadVariableOp_2:value:05ResNet/batch_normalization_6/ReadVariableOp_3:value:0FResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( У
ResNet/activation_6/Relu_1Relu3ResNet/batch_normalization_6/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ У
ResNet/activation_4/Relu_1Relu3ResNet/batch_normalization_4/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ л
ResNet/add_1/add_1AddV2(ResNet/activation_6/Relu_1:activations:0(ResNet/activation_4/Relu_1:activations:0*
T0*1
_output_shapes
:         ЎЎ Ю
-ResNet/batch_normalization_7/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0а
-ResNet/batch_normalization_7/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0└
>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0─
@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0с
/ResNet/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3ResNet/add_1/add_1:z:05ResNet/batch_normalization_7/ReadVariableOp_2:value:05ResNet/batch_normalization_7/ReadVariableOp_3:value:0FResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЎЎ : : : : :*
epsilon%oГ:*
is_training( У
ResNet/activation_7/Relu_1Relu3ResNet/batch_normalization_7/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:         ЎЎ Ю
'ResNet/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0с
ResNet/conv2d_6/Conv2D_1Conv2D(ResNet/activation_7/Relu_1:activations:0/ResNet/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ
*
paddingSAME*
strides
Ф
(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╡
ResNet/conv2d_6/BiasAdd_1BiasAdd!ResNet/conv2d_6/Conv2D_1:output:00ResNet/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЎЎ
Д
ResNet/conv2d_6/Softmax_1Softmax"ResNet/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:         ЎЎ
Ы
Jimage_projective_transform_layer_1/ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Н
Himage_projective_transform_layer_1/ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
=image_projective_transform_layer_1/ImageProjectiveTransformV3ImageProjectiveTransformV3#ResNet/conv2d_6/Softmax_1:softmax:0-random_affine_transform_params/stack:output:0Simage_projective_transform_layer_1/ImageProjectiveTransformV3/output_shape:output:0Qimage_projective_transform_layer_1/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         АА
*
dtype0*
interpolation
BILINEARА
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ё
"tf.compat.v1.transpose_1/transpose	TransposeRimage_projective_transform_layer_1/ImageProjectiveTransformV3:transformed_images:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА         
┌
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"
                  
*
paddingVALID*
strides
Е
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"              З
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ў
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:

^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АAЦ
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:

`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:

x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       й
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:

У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:

`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:

s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         й
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        н
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:

]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:

С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:

^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:

z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:

{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"    ■   С
tf.math.reduce_sum_2/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_2/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
tf.math.reduce_mean/RankConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :│
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: К
IdentityIdentity!ResNet/conv2d_6/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+                           
a

Identity_1Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: ╚%
NoOpNoOp;^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp=^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*^ResNet/batch_normalization/ReadVariableOp,^ResNet/batch_normalization/ReadVariableOp_1,^ResNet/batch_normalization/ReadVariableOp_2,^ResNet/batch_normalization/ReadVariableOp_3=^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_1/ReadVariableOp.^ResNet/batch_normalization_1/ReadVariableOp_1.^ResNet/batch_normalization_1/ReadVariableOp_2.^ResNet/batch_normalization_1/ReadVariableOp_3=^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_2/ReadVariableOp.^ResNet/batch_normalization_2/ReadVariableOp_1.^ResNet/batch_normalization_2/ReadVariableOp_2.^ResNet/batch_normalization_2/ReadVariableOp_3=^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_3/ReadVariableOp.^ResNet/batch_normalization_3/ReadVariableOp_1.^ResNet/batch_normalization_3/ReadVariableOp_2.^ResNet/batch_normalization_3/ReadVariableOp_3=^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_4/ReadVariableOp.^ResNet/batch_normalization_4/ReadVariableOp_1.^ResNet/batch_normalization_4/ReadVariableOp_2.^ResNet/batch_normalization_4/ReadVariableOp_3=^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_5/ReadVariableOp.^ResNet/batch_normalization_5/ReadVariableOp_1.^ResNet/batch_normalization_5/ReadVariableOp_2.^ResNet/batch_normalization_5/ReadVariableOp_3=^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_6/ReadVariableOp.^ResNet/batch_normalization_6/ReadVariableOp_1.^ResNet/batch_normalization_6/ReadVariableOp_2.^ResNet/batch_normalization_6/ReadVariableOp_3=^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_7/ReadVariableOp.^ResNet/batch_normalization_7/ReadVariableOp_1.^ResNet/batch_normalization_7/ReadVariableOp_2.^ResNet/batch_normalization_7/ReadVariableOp_3%^ResNet/conv2d/BiasAdd/ReadVariableOp'^ResNet/conv2d/BiasAdd_1/ReadVariableOp$^ResNet/conv2d/Conv2D/ReadVariableOp&^ResNet/conv2d/Conv2D_1/ReadVariableOp'^ResNet/conv2d_1/BiasAdd/ReadVariableOp)^ResNet/conv2d_1/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_1/Conv2D/ReadVariableOp(^ResNet/conv2d_1/Conv2D_1/ReadVariableOp'^ResNet/conv2d_2/BiasAdd/ReadVariableOp)^ResNet/conv2d_2/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_2/Conv2D/ReadVariableOp(^ResNet/conv2d_2/Conv2D_1/ReadVariableOp'^ResNet/conv2d_3/BiasAdd/ReadVariableOp)^ResNet/conv2d_3/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_3/Conv2D/ReadVariableOp(^ResNet/conv2d_3/Conv2D_1/ReadVariableOp'^ResNet/conv2d_4/BiasAdd/ReadVariableOp)^ResNet/conv2d_4/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_4/Conv2D/ReadVariableOp(^ResNet/conv2d_4/Conv2D_1/ReadVariableOp'^ResNet/conv2d_5/BiasAdd/ReadVariableOp)^ResNet/conv2d_5/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_5/Conv2D/ReadVariableOp(^ResNet/conv2d_5/Conv2D_1/ReadVariableOp'^ResNet/conv2d_6/BiasAdd/ReadVariableOp)^ResNet/conv2d_6/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_6/Conv2D/ReadVariableOp(^ResNet/conv2d_6/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp2|
<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_12|
<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp2А
>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_12V
)ResNet/batch_normalization/ReadVariableOp)ResNet/batch_normalization/ReadVariableOp2Z
+ResNet/batch_normalization/ReadVariableOp_1+ResNet/batch_normalization/ReadVariableOp_12Z
+ResNet/batch_normalization/ReadVariableOp_2+ResNet/batch_normalization/ReadVariableOp_22Z
+ResNet/batch_normalization/ReadVariableOp_3+ResNet/batch_normalization/ReadVariableOp_32|
<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_1/ReadVariableOp+ResNet/batch_normalization_1/ReadVariableOp2^
-ResNet/batch_normalization_1/ReadVariableOp_1-ResNet/batch_normalization_1/ReadVariableOp_12^
-ResNet/batch_normalization_1/ReadVariableOp_2-ResNet/batch_normalization_1/ReadVariableOp_22^
-ResNet/batch_normalization_1/ReadVariableOp_3-ResNet/batch_normalization_1/ReadVariableOp_32|
<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_2/ReadVariableOp+ResNet/batch_normalization_2/ReadVariableOp2^
-ResNet/batch_normalization_2/ReadVariableOp_1-ResNet/batch_normalization_2/ReadVariableOp_12^
-ResNet/batch_normalization_2/ReadVariableOp_2-ResNet/batch_normalization_2/ReadVariableOp_22^
-ResNet/batch_normalization_2/ReadVariableOp_3-ResNet/batch_normalization_2/ReadVariableOp_32|
<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_3/ReadVariableOp+ResNet/batch_normalization_3/ReadVariableOp2^
-ResNet/batch_normalization_3/ReadVariableOp_1-ResNet/batch_normalization_3/ReadVariableOp_12^
-ResNet/batch_normalization_3/ReadVariableOp_2-ResNet/batch_normalization_3/ReadVariableOp_22^
-ResNet/batch_normalization_3/ReadVariableOp_3-ResNet/batch_normalization_3/ReadVariableOp_32|
<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_4/ReadVariableOp+ResNet/batch_normalization_4/ReadVariableOp2^
-ResNet/batch_normalization_4/ReadVariableOp_1-ResNet/batch_normalization_4/ReadVariableOp_12^
-ResNet/batch_normalization_4/ReadVariableOp_2-ResNet/batch_normalization_4/ReadVariableOp_22^
-ResNet/batch_normalization_4/ReadVariableOp_3-ResNet/batch_normalization_4/ReadVariableOp_32|
<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_5/ReadVariableOp+ResNet/batch_normalization_5/ReadVariableOp2^
-ResNet/batch_normalization_5/ReadVariableOp_1-ResNet/batch_normalization_5/ReadVariableOp_12^
-ResNet/batch_normalization_5/ReadVariableOp_2-ResNet/batch_normalization_5/ReadVariableOp_22^
-ResNet/batch_normalization_5/ReadVariableOp_3-ResNet/batch_normalization_5/ReadVariableOp_32|
<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_6/ReadVariableOp+ResNet/batch_normalization_6/ReadVariableOp2^
-ResNet/batch_normalization_6/ReadVariableOp_1-ResNet/batch_normalization_6/ReadVariableOp_12^
-ResNet/batch_normalization_6/ReadVariableOp_2-ResNet/batch_normalization_6/ReadVariableOp_22^
-ResNet/batch_normalization_6/ReadVariableOp_3-ResNet/batch_normalization_6/ReadVariableOp_32|
<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2А
>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12А
>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp2Д
@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12Z
+ResNet/batch_normalization_7/ReadVariableOp+ResNet/batch_normalization_7/ReadVariableOp2^
-ResNet/batch_normalization_7/ReadVariableOp_1-ResNet/batch_normalization_7/ReadVariableOp_12^
-ResNet/batch_normalization_7/ReadVariableOp_2-ResNet/batch_normalization_7/ReadVariableOp_22^
-ResNet/batch_normalization_7/ReadVariableOp_3-ResNet/batch_normalization_7/ReadVariableOp_32L
$ResNet/conv2d/BiasAdd/ReadVariableOp$ResNet/conv2d/BiasAdd/ReadVariableOp2P
&ResNet/conv2d/BiasAdd_1/ReadVariableOp&ResNet/conv2d/BiasAdd_1/ReadVariableOp2J
#ResNet/conv2d/Conv2D/ReadVariableOp#ResNet/conv2d/Conv2D/ReadVariableOp2N
%ResNet/conv2d/Conv2D_1/ReadVariableOp%ResNet/conv2d/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_1/BiasAdd/ReadVariableOp&ResNet/conv2d_1/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_1/BiasAdd_1/ReadVariableOp(ResNet/conv2d_1/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_1/Conv2D/ReadVariableOp%ResNet/conv2d_1/Conv2D/ReadVariableOp2R
'ResNet/conv2d_1/Conv2D_1/ReadVariableOp'ResNet/conv2d_1/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_2/BiasAdd/ReadVariableOp&ResNet/conv2d_2/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_2/BiasAdd_1/ReadVariableOp(ResNet/conv2d_2/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_2/Conv2D/ReadVariableOp%ResNet/conv2d_2/Conv2D/ReadVariableOp2R
'ResNet/conv2d_2/Conv2D_1/ReadVariableOp'ResNet/conv2d_2/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_3/BiasAdd/ReadVariableOp&ResNet/conv2d_3/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_3/BiasAdd_1/ReadVariableOp(ResNet/conv2d_3/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_3/Conv2D/ReadVariableOp%ResNet/conv2d_3/Conv2D/ReadVariableOp2R
'ResNet/conv2d_3/Conv2D_1/ReadVariableOp'ResNet/conv2d_3/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_4/BiasAdd/ReadVariableOp&ResNet/conv2d_4/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_4/BiasAdd_1/ReadVariableOp(ResNet/conv2d_4/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_4/Conv2D/ReadVariableOp%ResNet/conv2d_4/Conv2D/ReadVariableOp2R
'ResNet/conv2d_4/Conv2D_1/ReadVariableOp'ResNet/conv2d_4/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_5/BiasAdd/ReadVariableOp&ResNet/conv2d_5/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_5/BiasAdd_1/ReadVariableOp(ResNet/conv2d_5/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_5/Conv2D/ReadVariableOp%ResNet/conv2d_5/Conv2D/ReadVariableOp2R
'ResNet/conv2d_5/Conv2D_1/ReadVariableOp'ResNet/conv2d_5/Conv2D_1/ReadVariableOp2P
&ResNet/conv2d_6/BiasAdd/ReadVariableOp&ResNet/conv2d_6/BiasAdd/ReadVariableOp2T
(ResNet/conv2d_6/BiasAdd_1/ReadVariableOp(ResNet/conv2d_6/BiasAdd_1/ReadVariableOp2N
%ResNet/conv2d_6/Conv2D/ReadVariableOp%ResNet/conv2d_6/Conv2D/ReadVariableOp2R
'ResNet/conv2d_6/Conv2D_1/ReadVariableOp'ResNet/conv2d_6/Conv2D_1/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391945

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ф
J
.__inference_activation_1_layer_call_fn_1391684

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1387402z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
т
Й
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1391573

inputs

transforms
identityx
'ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"v  v  j
%ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
ImageProjectiveTransformV3ImageProjectiveTransformV3inputs
transforms0ImageProjectiveTransformV3/output_shape:output:0.ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         ЎЎ*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:         ЎЎ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:+                           :         :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:SO
'
_output_shapes
:         
$
_user_specified_name
transforms
═
Э
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391771

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ў
■
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1392332

inputs8
conv2d_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
p
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           
z
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+                           
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
│
c
G__inference_activation_layer_call_and_return_conditional_losses_1391871

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
т
Й
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1388706

inputs

transforms
identityx
'ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"v  v  j
%ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
ImageProjectiveTransformV3ImageProjectiveTransformV3inputs
transforms0ImageProjectiveTransformV3/output_shape:output:0.ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         ЎЎ*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:         ЎЎ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:+                           :         :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:SO
'
_output_shapes
:         
$
_user_specified_name
transforms
╡
e
I__inference_activation_7_layer_call_and_return_conditional_losses_1387610

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Г
■
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1391708

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0л
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╡
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1387490

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                           t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╕Б
√
C__inference_ResNet_layer_call_and_return_conditional_losses_1387630

inputs*
conv2d_1_1387383:
conv2d_1_1387385:+
batch_normalization_1_1387388:+
batch_normalization_1_1387390:+
batch_normalization_1_1387392:+
batch_normalization_1_1387394:(
conv2d_1387415:
conv2d_1387417:*
conv2d_2_1387431:
conv2d_2_1387433:)
batch_normalization_1387436:)
batch_normalization_1387438:)
batch_normalization_1387440:)
batch_normalization_1387442:+
batch_normalization_2_1387445:+
batch_normalization_2_1387447:+
batch_normalization_2_1387449:+
batch_normalization_2_1387451:+
batch_normalization_3_1387476:+
batch_normalization_3_1387478:+
batch_normalization_3_1387480:+
batch_normalization_3_1387482:*
conv2d_4_1387503: 
conv2d_4_1387505: +
batch_normalization_5_1387508: +
batch_normalization_5_1387510: +
batch_normalization_5_1387512: +
batch_normalization_5_1387514: *
conv2d_3_1387535: 
conv2d_3_1387537: *
conv2d_5_1387551:  
conv2d_5_1387553: +
batch_normalization_4_1387556: +
batch_normalization_4_1387558: +
batch_normalization_4_1387560: +
batch_normalization_4_1387562: +
batch_normalization_6_1387565: +
batch_normalization_6_1387567: +
batch_normalization_6_1387569: +
batch_normalization_6_1387571: +
batch_normalization_7_1387596: +
batch_normalization_7_1387598: +
batch_normalization_7_1387600: +
batch_normalization_7_1387602: *
conv2d_6_1387624: 

conv2d_6_1387626:

identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallТ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_1387383conv2d_1_1387385*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1387382л
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1387388batch_normalization_1_1387390batch_normalization_1_1387392batch_normalization_1_1387394*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1386875Р
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1387402К
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1387415conv2d_1387417*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1387414▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1387431conv2d_2_1387433*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1387430Э
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1387436batch_normalization_1387438batch_normalization_1387440batch_normalization_1387442*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1387003л
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1387445batch_normalization_2_1387447batch_normalization_2_1387449batch_normalization_2_1387451*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1386939Р
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1387459К
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1387466У
add/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_1387474Ю
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_1387476batch_normalization_3_1387478batch_normalization_3_1387480batch_normalization_3_1387482*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1387067Р
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1387490▒
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1387503conv2d_4_1387505*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1387502л
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_1387508batch_normalization_5_1387510batch_normalization_5_1387512batch_normalization_5_1387514*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1387131Р
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1387522▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_3_1387535conv2d_3_1387537*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1387534▒
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_5_1387551conv2d_5_1387553*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1387550л
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1387556batch_normalization_4_1387558batch_normalization_4_1387560batch_normalization_4_1387562*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1387259л
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_1387565batch_normalization_6_1387567batch_normalization_6_1387569batch_normalization_6_1387571*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1387195Р
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1387579Р
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1387586Щ
add_1/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:0%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_1387594а
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_1387596batch_normalization_7_1387598batch_normalization_7_1387600batch_normalization_7_1387602*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387323Р
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_1387610▒
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0conv2d_6_1387624conv2d_6_1387626*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1387623Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
╖
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
J
"__inference__update_step_xla_16823
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
л
J
"__inference__update_step_xla_16798
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
─
Л
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1391587

inputs

transforms
identityx
'ImageProjectiveTransformV3/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      j
%ImageProjectiveTransformV3/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Л
ImageProjectiveTransformV3ImageProjectiveTransformV3inputs
transforms0ImageProjectiveTransformV3/output_shape:output:0.ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:         АА
*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:         АА
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         ЎЎ
:         :Y U
1
_output_shapes
:         ЎЎ

 
_user_specified_nameinputs:SO
'
_output_shapes
:         
$
_user_specified_name
transforms
═
Э
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392018

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
║
Я
*__inference_conv2d_5_layer_call_fn_1392055

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1387550Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
у
F
*__inference_add_loss_layer_call_fn_1391593

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_add_loss_layer_call_and_return_conditional_losses_1388810O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392146

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╡
e
I__inference_activation_5_layer_call_and_return_conditional_losses_1387522

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╡
e
I__inference_activation_5_layer_call_and_return_conditional_losses_1392046

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+                            t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                            :i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_7_layer_call_fn_1392266

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1387354Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ц─
╖3
 __inference__traced_save_1392685
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
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop5
1savev2_adam_m_conv2d_1_kernel_read_readvariableop5
1savev2_adam_v_conv2d_1_kernel_read_readvariableop3
/savev2_adam_m_conv2d_1_bias_read_readvariableop3
/savev2_adam_v_conv2d_1_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_1_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_1_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_1_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_1_beta_read_readvariableop5
1savev2_adam_m_conv2d_2_kernel_read_readvariableop5
1savev2_adam_v_conv2d_2_kernel_read_readvariableop3
/savev2_adam_m_conv2d_2_bias_read_readvariableop3
/savev2_adam_v_conv2d_2_bias_read_readvariableop3
/savev2_adam_m_conv2d_kernel_read_readvariableop3
/savev2_adam_v_conv2d_kernel_read_readvariableop1
-savev2_adam_m_conv2d_bias_read_readvariableop1
-savev2_adam_v_conv2d_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_2_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_2_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_2_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_2_beta_read_readvariableop?
;savev2_adam_m_batch_normalization_gamma_read_readvariableop?
;savev2_adam_v_batch_normalization_gamma_read_readvariableop>
:savev2_adam_m_batch_normalization_beta_read_readvariableop>
:savev2_adam_v_batch_normalization_beta_read_readvariableopA
=savev2_adam_m_batch_normalization_3_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_3_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_3_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_3_beta_read_readvariableop5
1savev2_adam_m_conv2d_4_kernel_read_readvariableop5
1savev2_adam_v_conv2d_4_kernel_read_readvariableop3
/savev2_adam_m_conv2d_4_bias_read_readvariableop3
/savev2_adam_v_conv2d_4_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_5_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_5_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_5_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_5_beta_read_readvariableop5
1savev2_adam_m_conv2d_5_kernel_read_readvariableop5
1savev2_adam_v_conv2d_5_kernel_read_readvariableop3
/savev2_adam_m_conv2d_5_bias_read_readvariableop3
/savev2_adam_v_conv2d_5_bias_read_readvariableop5
1savev2_adam_m_conv2d_3_kernel_read_readvariableop5
1savev2_adam_v_conv2d_3_kernel_read_readvariableop3
/savev2_adam_m_conv2d_3_bias_read_readvariableop3
/savev2_adam_v_conv2d_3_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_6_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_6_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_6_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_6_beta_read_readvariableopA
=savev2_adam_m_batch_normalization_4_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_4_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_4_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_4_beta_read_readvariableopA
=savev2_adam_m_batch_normalization_7_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_7_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_7_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_7_beta_read_readvariableop5
1savev2_adam_m_conv2d_6_kernel_read_readvariableop5
1savev2_adam_v_conv2d_6_kernel_read_readvariableop3
/savev2_adam_m_conv2d_6_bias_read_readvariableop3
/savev2_adam_v_conv2d_6_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╪)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*Б)
valueў(BЇ(oB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╬
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:o*
dtype0*є
valueщBцoB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╪1
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop1savev2_adam_m_conv2d_1_kernel_read_readvariableop1savev2_adam_v_conv2d_1_kernel_read_readvariableop/savev2_adam_m_conv2d_1_bias_read_readvariableop/savev2_adam_v_conv2d_1_bias_read_readvariableop=savev2_adam_m_batch_normalization_1_gamma_read_readvariableop=savev2_adam_v_batch_normalization_1_gamma_read_readvariableop<savev2_adam_m_batch_normalization_1_beta_read_readvariableop<savev2_adam_v_batch_normalization_1_beta_read_readvariableop1savev2_adam_m_conv2d_2_kernel_read_readvariableop1savev2_adam_v_conv2d_2_kernel_read_readvariableop/savev2_adam_m_conv2d_2_bias_read_readvariableop/savev2_adam_v_conv2d_2_bias_read_readvariableop/savev2_adam_m_conv2d_kernel_read_readvariableop/savev2_adam_v_conv2d_kernel_read_readvariableop-savev2_adam_m_conv2d_bias_read_readvariableop-savev2_adam_v_conv2d_bias_read_readvariableop=savev2_adam_m_batch_normalization_2_gamma_read_readvariableop=savev2_adam_v_batch_normalization_2_gamma_read_readvariableop<savev2_adam_m_batch_normalization_2_beta_read_readvariableop<savev2_adam_v_batch_normalization_2_beta_read_readvariableop;savev2_adam_m_batch_normalization_gamma_read_readvariableop;savev2_adam_v_batch_normalization_gamma_read_readvariableop:savev2_adam_m_batch_normalization_beta_read_readvariableop:savev2_adam_v_batch_normalization_beta_read_readvariableop=savev2_adam_m_batch_normalization_3_gamma_read_readvariableop=savev2_adam_v_batch_normalization_3_gamma_read_readvariableop<savev2_adam_m_batch_normalization_3_beta_read_readvariableop<savev2_adam_v_batch_normalization_3_beta_read_readvariableop1savev2_adam_m_conv2d_4_kernel_read_readvariableop1savev2_adam_v_conv2d_4_kernel_read_readvariableop/savev2_adam_m_conv2d_4_bias_read_readvariableop/savev2_adam_v_conv2d_4_bias_read_readvariableop=savev2_adam_m_batch_normalization_5_gamma_read_readvariableop=savev2_adam_v_batch_normalization_5_gamma_read_readvariableop<savev2_adam_m_batch_normalization_5_beta_read_readvariableop<savev2_adam_v_batch_normalization_5_beta_read_readvariableop1savev2_adam_m_conv2d_5_kernel_read_readvariableop1savev2_adam_v_conv2d_5_kernel_read_readvariableop/savev2_adam_m_conv2d_5_bias_read_readvariableop/savev2_adam_v_conv2d_5_bias_read_readvariableop1savev2_adam_m_conv2d_3_kernel_read_readvariableop1savev2_adam_v_conv2d_3_kernel_read_readvariableop/savev2_adam_m_conv2d_3_bias_read_readvariableop/savev2_adam_v_conv2d_3_bias_read_readvariableop=savev2_adam_m_batch_normalization_6_gamma_read_readvariableop=savev2_adam_v_batch_normalization_6_gamma_read_readvariableop<savev2_adam_m_batch_normalization_6_beta_read_readvariableop<savev2_adam_v_batch_normalization_6_beta_read_readvariableop=savev2_adam_m_batch_normalization_4_gamma_read_readvariableop=savev2_adam_v_batch_normalization_4_gamma_read_readvariableop<savev2_adam_m_batch_normalization_4_beta_read_readvariableop<savev2_adam_v_batch_normalization_4_beta_read_readvariableop=savev2_adam_m_batch_normalization_7_gamma_read_readvariableop=savev2_adam_v_batch_normalization_7_gamma_read_readvariableop<savev2_adam_m_batch_normalization_7_beta_read_readvariableop<savev2_adam_v_batch_normalization_7_beta_read_readvariableop1savev2_adam_m_conv2d_6_kernel_read_readvariableop1savev2_adam_v_conv2d_6_kernel_read_readvariableop/savev2_adam_m_conv2d_6_bias_read_readvariableop/savev2_adam_v_conv2d_6_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *}
dtypess
q2o	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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

identity_1Identity_1:output:0*Щ
_input_shapesЗ
Д: ::::::::::::::::::::::: : : : : : :  : : : : : : : : : : : : : : : : 
:
: : ::::::::::::::::::::::::::::: : : : : : : : :  :  : : : : : : : : : : : : : : : : : : : 
: 
:
:
: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: :,-(
&
_output_shapes
: 
: .

_output_shapes
:
:/

_output_shapes
: :0

_output_shapes
: :,1(
&
_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
::,9(
&
_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
::,=(
&
_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
:: L

_output_shapes
::,M(
&
_output_shapes
: :,N(
&
_output_shapes
: : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: : R

_output_shapes
: : S

_output_shapes
: : T

_output_shapes
: :,U(
&
_output_shapes
:  :,V(
&
_output_shapes
:  : W

_output_shapes
: : X

_output_shapes
: :,Y(
&
_output_shapes
: :,Z(
&
_output_shapes
: : [

_output_shapes
: : \

_output_shapes
: : ]

_output_shapes
: : ^

_output_shapes
: : _

_output_shapes
: : `

_output_shapes
: : a

_output_shapes
: : b

_output_shapes
: : c

_output_shapes
: : d

_output_shapes
: : e

_output_shapes
: : f

_output_shapes
: : g

_output_shapes
: : h

_output_shapes
: :,i(
&
_output_shapes
: 
:,j(
&
_output_shapes
: 
: k

_output_shapes
:
: l

_output_shapes
:
:m

_output_shapes
: :n

_output_shapes
: :o

_output_shapes
: 
л
J
"__inference__update_step_xla_16808
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ъ
╒

(__inference_ResNet_layer_call_fn_1388269
input_2!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21: 

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: $

unknown_29:  

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42: $

unknown_43: 


unknown_44:

identityИвStatefulPartitionedCall╟
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_ResNet_layer_call_and_return_conditional_losses_1388077Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+                           
!
_user_specified_name	input_2
й╥
▄(
C__inference_ResNet_layer_call_and_return_conditional_losses_1391301

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:;
-batch_normalization_3_readvariableop_resource:=
/batch_normalization_3_readvariableop_1_resource:L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: ;
-batch_normalization_5_readvariableop_resource: =
/batch_normalization_5_readvariableop_1_resource: L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource:  6
(conv2d_5_biasadd_readvariableop_resource: ;
-batch_normalization_4_readvariableop_resource: =
/batch_normalization_4_readvariableop_1_resource: L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_6_conv2d_readvariableop_resource: 
6
(conv2d_6_biasadd_readvariableop_resource:

identityИв3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1в5batch_normalization_3/FusedBatchNormV3/ReadVariableOpв7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_3/ReadVariableOpв&batch_normalization_3/ReadVariableOp_1в5batch_normalization_4/FusedBatchNormV3/ReadVariableOpв7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_4/ReadVariableOpв&batch_normalization_4/ReadVariableOp_1в5batch_normalization_5/FusedBatchNormV3/ReadVariableOpв7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_5/ReadVariableOpв&batch_normalization_5/ReadVariableOp_1в5batch_normalization_6/FusedBatchNormV3/ReadVariableOpв7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_6/ReadVariableOpв&batch_normalization_6/ReadVariableOp_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1вconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOpвconv2d_4/BiasAdd/ReadVariableOpвconv2d_4/Conv2D/ReadVariableOpвconv2d_5/BiasAdd/ReadVariableOpвconv2d_5/Conv2D/ReadVariableOpвconv2d_6/BiasAdd/ReadVariableOpвconv2d_6/Conv2D/ReadVariableOpО
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╜
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0к
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╔
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( С
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╣
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╓
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0к
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0м
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╜
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╔
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( С
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Н
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           Ь
add/addAddV2activation_2/Relu:activations:0activation/Relu:activations:0*
T0*A
_output_shapes/
-:+                           О
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╗
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3add/add:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( С
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           О
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╓
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Д
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            О
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╔
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( С
activation_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            О
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╓
conv2d_3/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Д
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            О
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0╓
conv2d_5/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            О
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╔
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( О
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╔
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( С
activation_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            С
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            а
	add_1/addAddV2activation_6/Relu:activations:0activation_4/Relu:activations:0*
T0*A
_output_shapes/
-:+                            О
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╜
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3add_1/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( С
activation_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: 
*
dtype0╓
conv2d_6/Conv2DConv2Dactivation_7/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0к
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           
В
conv2d_6/SoftmaxSoftmaxconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           
Г
IdentityIdentityconv2d_6/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+                           
Я
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+                           : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
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
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12>
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
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▌
serving_default╔
U
input_1J
serving_default_input_1:0+                           T
ResNetJ
StatefulPartitionedCall:0+                           
tensorflow/serving/predict:ХФ
Е
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_default_save_signature
$	optimizer
%loss
&
signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ф
'layer-0
(layer_with_weights-0
(layer-1
)layer_with_weights-1
)layer-2
*layer-3
+layer_with_weights-2
+layer-4
,layer_with_weights-3
,layer-5
-layer_with_weights-4
-layer-6
.layer_with_weights-5
.layer-7
/layer-8
0layer-9
1layer-10
2layer_with_weights-6
2layer-11
3layer-12
4layer_with_weights-7
4layer-13
5layer_with_weights-8
5layer-14
6layer-15
7layer_with_weights-9
7layer-16
8layer_with_weights-10
8layer-17
9layer_with_weights-11
9layer-18
:layer_with_weights-12
:layer-19
;layer-20
<layer-21
=layer-22
>layer_with_weights-13
>layer-23
?layer-24
@layer_with_weights-14
@layer-25
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_network
е
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
е
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
(
S	keras_api"
_tf_keras_layer
е
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
(
Z	keras_api"
_tf_keras_layer
(
[	keras_api"
_tf_keras_layer
(
\	keras_api"
_tf_keras_layer
(
]	keras_api"
_tf_keras_layer
(
^	keras_api"
_tf_keras_layer
(
_	keras_api"
_tf_keras_layer
(
`	keras_api"
_tf_keras_layer
(
a	keras_api"
_tf_keras_layer
(
b	keras_api"
_tf_keras_layer
(
c	keras_api"
_tf_keras_layer
(
d	keras_api"
_tf_keras_layer
(
e	keras_api"
_tf_keras_layer
(
f	keras_api"
_tf_keras_layer
(
g	keras_api"
_tf_keras_layer
(
h	keras_api"
_tf_keras_layer
(
i	keras_api"
_tf_keras_layer
(
j	keras_api"
_tf_keras_layer
(
k	keras_api"
_tf_keras_layer
(
l	keras_api"
_tf_keras_layer
(
m	keras_api"
_tf_keras_layer
(
n	keras_api"
_tf_keras_layer
е
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
й
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
П26
Р27
С28
Т29
У30
Ф31
Х32
Ц33
Ч34
Ш35
Щ36
Ъ37
Ы38
Ь39
Э40
Ю41
Я42
а43
б44
в45"
trackable_list_wrapper
Ы
u0
v1
w2
x3
{4
|5
}6
~7
8
А9
Г10
Д11
З12
И13
Л14
М15
Н16
О17
С18
Т19
У20
Ф21
Х22
Ц23
Щ24
Ъ25
Э26
Ю27
б28
в29"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
#_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
┘
иtrace_0
йtrace_1
кtrace_2
лtrace_32ц
'__inference_model_layer_call_fn_1388911
'__inference_model_layer_call_fn_1390012
'__inference_model_layer_call_fn_1390110
'__inference_model_layer_call_fn_1389427┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0zйtrace_1zкtrace_2zлtrace_3
┼
мtrace_0
нtrace_1
оtrace_2
пtrace_32╥
B__inference_model_layer_call_and_return_conditional_losses_1390524
B__inference_model_layer_call_and_return_conditional_losses_1390938
B__inference_model_layer_call_and_return_conditional_losses_1389620
B__inference_model_layer_call_and_return_conditional_losses_1389813┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0zнtrace_1zоtrace_2zпtrace_3
═B╩
"__inference__wrapped_model_1386853input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
г
░
_variables
▒_iterations
▓_learning_rate
│_index_dict
┤
_momentums
╡_velocities
╢_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
-
╖serving_default"
signature_map
"
_tf_keras_input_layer
ф
╕	variables
╣trainable_variables
║regularization_losses
╗	keras_api
╝__call__
+╜&call_and_return_all_conditional_losses

ukernel
vbias
!╛_jit_compiled_convolution_op"
_tf_keras_layer
ё
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
├__call__
+─&call_and_return_all_conditional_losses
	┼axis
	wgamma
xbeta
ymoving_mean
zmoving_variance"
_tf_keras_layer
л
╞	variables
╟trainable_variables
╚regularization_losses
╔	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
╠	variables
═trainable_variables
╬regularization_losses
╧	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses

{kernel
|bias
!╥_jit_compiled_convolution_op"
_tf_keras_layer
ф
╙	variables
╘trainable_variables
╒regularization_losses
╓	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses

}kernel
~bias
!┘_jit_compiled_convolution_op"
_tf_keras_layer
Ї
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
▐__call__
+▀&call_and_return_all_conditional_losses
	рaxis
	gamma
	Аbeta
Бmoving_mean
Вmoving_variance"
_tf_keras_layer
ї
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
	чaxis

Гgamma
	Дbeta
Еmoving_mean
Жmoving_variance"
_tf_keras_layer
л
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
л
ю	variables
яtrainable_variables
Ёregularization_losses
ё	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Ї	variables
їtrainable_variables
Ўregularization_losses
ў	keras_api
°__call__
+∙&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
·	variables
√trainable_variables
№regularization_losses
¤	keras_api
■__call__
+ &call_and_return_all_conditional_losses
	Аaxis

Зgamma
	Иbeta
Йmoving_mean
Кmoving_variance"
_tf_keras_layer
л
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses
Лkernel
	Мbias
!Н_jit_compiled_convolution_op"
_tf_keras_layer
ї
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses
	Фaxis

Нgamma
	Оbeta
Пmoving_mean
Рmoving_variance"
_tf_keras_layer
л
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
Сkernel
	Тbias
!б_jit_compiled_convolution_op"
_tf_keras_layer
ц
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
Уkernel
	Фbias
!и_jit_compiled_convolution_op"
_tf_keras_layer
ї
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses
	пaxis

Хgamma
	Цbeta
Чmoving_mean
Шmoving_variance"
_tf_keras_layer
ї
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
┤__call__
+╡&call_and_return_all_conditional_losses
	╢axis

Щgamma
	Ъbeta
Ыmoving_mean
Ьmoving_variance"
_tf_keras_layer
л
╖	variables
╕trainable_variables
╣regularization_losses
║	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"
_tf_keras_layer
л
╜	variables
╛trainable_variables
┐regularization_losses
└	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"
_tf_keras_layer
л
├	variables
─trainable_variables
┼regularization_losses
╞	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses
	╧axis

Эgamma
	Юbeta
Яmoving_mean
аmoving_variance"
_tf_keras_layer
л
╨	variables
╤trainable_variables
╥regularization_losses
╙	keras_api
╘__call__
+╒&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses
бkernel
	вbias
!▄_jit_compiled_convolution_op"
_tf_keras_layer
й
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
П26
Р27
С28
Т29
У30
Ф31
Х32
Ц33
Ч34
Ш35
Щ36
Ъ37
Ы38
Ь39
Э40
Ю41
Я42
а43
б44
в45"
trackable_list_wrapper
Ы
u0
v1
w2
x3
{4
|5
}6
~7
8
А9
Г10
Д11
З12
И13
Л14
М15
Н16
О17
С18
Т19
У20
Ф21
Х22
Ц23
Щ24
Ъ25
Э26
Ю27
б28
в29"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
▌
тtrace_0
уtrace_1
фtrace_2
хtrace_32ъ
(__inference_ResNet_layer_call_fn_1387725
(__inference_ResNet_layer_call_fn_1391035
(__inference_ResNet_layer_call_fn_1391132
(__inference_ResNet_layer_call_fn_1388269┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zтtrace_0zуtrace_1zфtrace_2zхtrace_3
╔
цtrace_0
чtrace_1
шtrace_2
щtrace_32╓
C__inference_ResNet_layer_call_and_return_conditional_losses_1391301
C__inference_ResNet_layer_call_and_return_conditional_losses_1391470
C__inference_ResNet_layer_call_and_return_conditional_losses_1388390
C__inference_ResNet_layer_call_and_return_conditional_losses_1388511┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0zчtrace_1zшtrace_2zщtrace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
Т
яtrace_02є
@__inference_random_affine_transform_params_layer_call_fn_1391477о
е▓б
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsв
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0
н
Ёtrace_02О
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1391559о
е▓б
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsв
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
┤
Ўtrace_02Х
B__inference_image_projective_transform_layer_layer_call_fn_1391565╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0
╧
ўtrace_02░
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1391573╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
╢
¤trace_02Ч
D__inference_image_projective_transform_layer_1_layer_call_fn_1391579╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z¤trace_0
╤
■trace_02▓
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1391587╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z■trace_0
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
Ё
Дtrace_02╤
*__inference_add_loss_layer_call_fn_1391593в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zДtrace_0
Л
Еtrace_02ь
E__inference_add_loss_layer_call_and_return_conditional_losses_1391598в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
):'2conv2d_1/kernel
:2conv2d_1/bias
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
):'2conv2d_2/kernel
:2conv2d_2/bias
':%2conv2d/kernel
:2conv2d/bias
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
):' 2conv2d_4/kernel
: 2conv2d_4/bias
):' 2batch_normalization_5/gamma
(:& 2batch_normalization_5/beta
1:/  (2!batch_normalization_5/moving_mean
5:3  (2%batch_normalization_5/moving_variance
):'  2conv2d_5/kernel
: 2conv2d_5/bias
):' 2conv2d_3/kernel
: 2conv2d_3/bias
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
):' 
2conv2d_6/kernel
:
2conv2d_6/bias
д
y0
z1
Б2
В3
Е4
Ж5
Й6
К7
П8
Р9
Ч10
Ш11
Ы12
Ь13
Я14
а15"
trackable_list_wrapper
Ў
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
27"
trackable_list_wrapper
(
Ж0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
∙BЎ
'__inference_model_layer_call_fn_1388911input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
'__inference_model_layer_call_fn_1390012inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
'__inference_model_layer_call_fn_1390110inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
'__inference_model_layer_call_fn_1389427input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
УBР
B__inference_model_layer_call_and_return_conditional_losses_1390524inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
УBР
B__inference_model_layer_call_and_return_conditional_losses_1390938inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
B__inference_model_layer_call_and_return_conditional_losses_1389620input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
B__inference_model_layer_call_and_return_conditional_losses_1389813input_1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╗
▒0
З1
И2
Й3
К4
Л5
М6
Н7
О8
П9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
Ч17
Ш18
Щ19
Ъ20
Ы21
Ь22
Э23
Ю24
Я25
а26
б27
в28
г29
д30
е31
ж32
з33
и34
й35
к36
л37
м38
н39
о40
п41
░42
▒43
▓44
│45
┤46
╡47
╢48
╖49
╕50
╣51
║52
╗53
╝54
╜55
╛56
┐57
└58
┴59
┬60"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
д
З0
Й1
Л2
Н3
П4
С5
У6
Х7
Ч8
Щ9
Ы10
Э11
Я12
б13
г14
е15
з16
й17
л18
н19
п20
▒21
│22
╡23
╖24
╣25
╗26
╜27
┐28
┴29"
trackable_list_wrapper
д
И0
К1
М2
О3
Р4
Т5
Ф6
Ц7
Ш8
Ъ9
Ь10
Ю11
а12
в13
д14
ж15
и16
к17
м18
о19
░20
▓21
┤22
╢23
╕24
║25
╝26
╛27
└28
┬29"
trackable_list_wrapper
ч
├trace_0
─trace_1
┼trace_2
╞trace_3
╟trace_4
╚trace_5
╔trace_6
╩trace_7
╦trace_8
╠trace_9
═trace_10
╬trace_11
╧trace_12
╨trace_13
╤trace_14
╥trace_15
╙trace_16
╘trace_17
╒trace_18
╓trace_19
╫trace_20
╪trace_21
┘trace_22
┌trace_23
█trace_24
▄trace_25
▌trace_26
▐trace_27
▀trace_28
рtrace_292Ї	
"__inference__update_step_xla_16748
"__inference__update_step_xla_16753
"__inference__update_step_xla_16758
"__inference__update_step_xla_16763
"__inference__update_step_xla_16768
"__inference__update_step_xla_16773
"__inference__update_step_xla_16778
"__inference__update_step_xla_16783
"__inference__update_step_xla_16788
"__inference__update_step_xla_16793
"__inference__update_step_xla_16798
"__inference__update_step_xla_16803
"__inference__update_step_xla_16808
"__inference__update_step_xla_16813
"__inference__update_step_xla_16818
"__inference__update_step_xla_16823
"__inference__update_step_xla_16828
"__inference__update_step_xla_16833
"__inference__update_step_xla_16838
"__inference__update_step_xla_16843
"__inference__update_step_xla_16848
"__inference__update_step_xla_16853
"__inference__update_step_xla_16858
"__inference__update_step_xla_16863
"__inference__update_step_xla_16868
"__inference__update_step_xla_16873
"__inference__update_step_xla_16878
"__inference__update_step_xla_16883
"__inference__update_step_xla_16888
"__inference__update_step_xla_16893╣
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0z├trace_0z─trace_1z┼trace_2z╞trace_3z╟trace_4z╚trace_5z╔trace_6z╩trace_7z╦trace_8z╠trace_9z═trace_10z╬trace_11z╧trace_12z╨trace_13z╤trace_14z╥trace_15z╙trace_16z╘trace_17z╒trace_18z╓trace_19z╫trace_20z╪trace_21z┘trace_22z┌trace_23z█trace_24z▄trace_25z▌trace_26z▐trace_27z▀trace_28zрtrace_29
╠B╔
%__inference_signature_wrapper_1389914input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
╕	variables
╣trainable_variables
║regularization_losses
╝__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
_generic_user_object
Ё
цtrace_02╤
*__inference_conv2d_1_layer_call_fn_1391607в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0
Л
чtrace_02ь
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1391617в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
w0
x1
y2
z3"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
┐	variables
└trainable_variables
┴regularization_losses
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
у
эtrace_0
юtrace_12и
7__inference_batch_normalization_1_layer_call_fn_1391630
7__inference_batch_normalization_1_layer_call_fn_1391643│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zэtrace_0zюtrace_1
Щ
яtrace_0
Ёtrace_12▐
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391661
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391679│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0zЁtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
╞	variables
╟trainable_variables
╚regularization_losses
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
Ї
Ўtrace_02╒
.__inference_activation_1_layer_call_fn_1391684в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0
П
ўtrace_02Ё
I__inference_activation_1_layer_call_and_return_conditional_losses_1391689в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
°non_trainable_variables
∙layers
·metrics
 √layer_regularization_losses
№layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
Ё
¤trace_02╤
*__inference_conv2d_2_layer_call_fn_1391698в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z¤trace_0
Л
■trace_02ь
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1391708в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z■trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
╙	variables
╘trainable_variables
╒regularization_losses
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses"
_generic_user_object
ю
Дtrace_02╧
(__inference_conv2d_layer_call_fn_1391717в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zДtrace_0
Й
Еtrace_02ъ
C__inference_conv2d_layer_call_and_return_conditional_losses_1391727в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
?
0
А1
Б2
В3"
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
┌	variables
█trainable_variables
▄regularization_losses
▐__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
у
Лtrace_0
Мtrace_12и
7__inference_batch_normalization_2_layer_call_fn_1391740
7__inference_batch_normalization_2_layer_call_fn_1391753│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЛtrace_0zМtrace_1
Щ
Нtrace_0
Оtrace_12▐
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391771
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391789│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0zОtrace_1
 "
trackable_list_wrapper
@
Г0
Д1
Е2
Ж3"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
▀
Фtrace_0
Хtrace_12д
5__inference_batch_normalization_layer_call_fn_1391802
5__inference_batch_normalization_layer_call_fn_1391815│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0zХtrace_1
Х
Цtrace_0
Чtrace_12┌
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391833
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391851│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЦtrace_0zЧtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
Ї
Эtrace_02╒
.__inference_activation_2_layer_call_fn_1391856в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЭtrace_0
П
Юtrace_02Ё
I__inference_activation_2_layer_call_and_return_conditional_losses_1391861в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЮtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
ю	variables
яtrainable_variables
Ёregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
Є
дtrace_02╙
,__inference_activation_layer_call_fn_1391866в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zдtrace_0
Н
еtrace_02ю
G__inference_activation_layer_call_and_return_conditional_losses_1391871в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
Ї	variables
їtrainable_variables
Ўregularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
ы
лtrace_02╠
%__inference_add_layer_call_fn_1391877в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0
Ж
мtrace_02ч
@__inference_add_layer_call_and_return_conditional_losses_1391883в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0
@
З0
И1
Й2
К3"
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
·	variables
√trainable_variables
№regularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
у
▓trace_0
│trace_12и
7__inference_batch_normalization_3_layer_call_fn_1391896
7__inference_batch_normalization_3_layer_call_fn_1391909│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0z│trace_1
Щ
┤trace_0
╡trace_12▐
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391927
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391945│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0z╡trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
Ї
╗trace_02╒
.__inference_activation_3_layer_call_fn_1391950в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0
П
╝trace_02Ё
I__inference_activation_3_layer_call_and_return_conditional_losses_1391955в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╝trace_0
0
Л0
М1"
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
Ё
┬trace_02╤
*__inference_conv2d_4_layer_call_fn_1391964в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0
Л
├trace_02ь
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1391974в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
@
Н0
О1
П2
Р3"
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
у
╔trace_0
╩trace_12и
7__inference_batch_normalization_5_layer_call_fn_1391987
7__inference_batch_normalization_5_layer_call_fn_1392000│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╔trace_0z╩trace_1
Щ
╦trace_0
╠trace_12▐
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392018
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392036│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0z╠trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
Ї
╥trace_02╒
.__inference_activation_5_layer_call_fn_1392041в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
П
╙trace_02Ё
I__inference_activation_5_layer_call_and_return_conditional_losses_1392046в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╙trace_0
0
С0
Т1"
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
Ё
┘trace_02╤
*__inference_conv2d_5_layer_call_fn_1392055в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
Л
┌trace_02ь
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1392065в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
0
У0
Ф1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
Ё
рtrace_02╤
*__inference_conv2d_3_layer_call_fn_1392074в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zрtrace_0
Л
сtrace_02ь
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1392084в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
@
Х0
Ц1
Ч2
Ш3"
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
у
чtrace_0
шtrace_12и
7__inference_batch_normalization_6_layer_call_fn_1392097
7__inference_batch_normalization_6_layer_call_fn_1392110│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0zшtrace_1
Щ
щtrace_0
ъtrace_12▐
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392128
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392146│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0zъtrace_1
 "
trackable_list_wrapper
@
Щ0
Ъ1
Ы2
Ь3"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
░	variables
▒trainable_variables
▓regularization_losses
┤__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
у
Ёtrace_0
ёtrace_12и
7__inference_batch_normalization_4_layer_call_fn_1392159
7__inference_batch_normalization_4_layer_call_fn_1392172│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0zёtrace_1
Щ
Єtrace_0
єtrace_12▐
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392190
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392208│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЄtrace_0zєtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Їnon_trainable_variables
їlayers
Ўmetrics
 ўlayer_regularization_losses
°layer_metrics
╖	variables
╕trainable_variables
╣regularization_losses
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
Ї
∙trace_02╒
.__inference_activation_6_layer_call_fn_1392213в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z∙trace_0
П
·trace_02Ё
I__inference_activation_6_layer_call_and_return_conditional_losses_1392218в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z·trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
√non_trainable_variables
№layers
¤metrics
 ■layer_regularization_losses
 layer_metrics
╜	variables
╛trainable_variables
┐regularization_losses
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
Ї
Аtrace_02╒
.__inference_activation_4_layer_call_fn_1392223в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zАtrace_0
П
Бtrace_02Ё
I__inference_activation_4_layer_call_and_return_conditional_losses_1392228в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zБtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
├	variables
─trainable_variables
┼regularization_losses
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
э
Зtrace_02╬
'__inference_add_1_layer_call_fn_1392234в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0
И
Иtrace_02щ
B__inference_add_1_layer_call_and_return_conditional_losses_1392240в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zИtrace_0
@
Э0
Ю1
Я2
а3"
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
у
Оtrace_0
Пtrace_12и
7__inference_batch_normalization_7_layer_call_fn_1392253
7__inference_batch_normalization_7_layer_call_fn_1392266│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zОtrace_0zПtrace_1
Щ
Рtrace_0
Сtrace_12▐
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392284
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392302│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zРtrace_0zСtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
╨	variables
╤trainable_variables
╥regularization_losses
╘__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses"
_generic_user_object
Ї
Чtrace_02╒
.__inference_activation_7_layer_call_fn_1392307в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЧtrace_0
П
Шtrace_02Ё
I__inference_activation_7_layer_call_and_return_conditional_losses_1392312в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
0
б0
в1"
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
Ё
Юtrace_02╤
*__inference_conv2d_6_layer_call_fn_1392321в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЮtrace_0
Л
Яtrace_02ь
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1392332в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЯtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
д
y0
z1
Б2
В3
Е4
Ж5
Й6
К7
П8
Р9
Ч10
Ш11
Ы12
Ь13
Я14
а15"
trackable_list_wrapper
ц
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?24
@25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
·Bў
(__inference_ResNet_layer_call_fn_1387725input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
(__inference_ResNet_layer_call_fn_1391035inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
(__inference_ResNet_layer_call_fn_1391132inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
(__inference_ResNet_layer_call_fn_1388269input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
C__inference_ResNet_layer_call_and_return_conditional_losses_1391301inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
C__inference_ResNet_layer_call_and_return_conditional_losses_1391470inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
C__inference_ResNet_layer_call_and_return_conditional_losses_1388390input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
C__inference_ResNet_layer_call_and_return_conditional_losses_1388511input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
¤B·
@__inference_random_affine_transform_params_layer_call_fn_1391477inp"о
е▓б
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsв
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1391559inp"о
е▓б
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsв
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
оBл
B__inference_image_projective_transform_layer_layer_call_fn_1391565inputs
transforms"╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╔B╞
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1391573inputs
transforms"╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
░Bн
D__inference_image_projective_transform_layer_1_layer_call_fn_1391579inputs
transforms"╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╦B╚
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1391587inputs
transforms"╬
┼▓┴
FullArgSpec>
args6Ъ3
jself
jinputs
j
transforms
jWIDTH
jHEIGHT
varargs
 
varkw
 
defaultsв

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_add_loss_layer_call_fn_1391593inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_add_loss_layer_call_and_return_conditional_losses_1391598inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
а	variables
б	keras_api

вtotal

гcount"
_tf_keras_metric
.:,2Adam/m/conv2d_1/kernel
.:,2Adam/v/conv2d_1/kernel
 :2Adam/m/conv2d_1/bias
 :2Adam/v/conv2d_1/bias
.:,2"Adam/m/batch_normalization_1/gamma
.:,2"Adam/v/batch_normalization_1/gamma
-:+2!Adam/m/batch_normalization_1/beta
-:+2!Adam/v/batch_normalization_1/beta
.:,2Adam/m/conv2d_2/kernel
.:,2Adam/v/conv2d_2/kernel
 :2Adam/m/conv2d_2/bias
 :2Adam/v/conv2d_2/bias
,:*2Adam/m/conv2d/kernel
,:*2Adam/v/conv2d/kernel
:2Adam/m/conv2d/bias
:2Adam/v/conv2d/bias
.:,2"Adam/m/batch_normalization_2/gamma
.:,2"Adam/v/batch_normalization_2/gamma
-:+2!Adam/m/batch_normalization_2/beta
-:+2!Adam/v/batch_normalization_2/beta
,:*2 Adam/m/batch_normalization/gamma
,:*2 Adam/v/batch_normalization/gamma
+:)2Adam/m/batch_normalization/beta
+:)2Adam/v/batch_normalization/beta
.:,2"Adam/m/batch_normalization_3/gamma
.:,2"Adam/v/batch_normalization_3/gamma
-:+2!Adam/m/batch_normalization_3/beta
-:+2!Adam/v/batch_normalization_3/beta
.:, 2Adam/m/conv2d_4/kernel
.:, 2Adam/v/conv2d_4/kernel
 : 2Adam/m/conv2d_4/bias
 : 2Adam/v/conv2d_4/bias
.:, 2"Adam/m/batch_normalization_5/gamma
.:, 2"Adam/v/batch_normalization_5/gamma
-:+ 2!Adam/m/batch_normalization_5/beta
-:+ 2!Adam/v/batch_normalization_5/beta
.:,  2Adam/m/conv2d_5/kernel
.:,  2Adam/v/conv2d_5/kernel
 : 2Adam/m/conv2d_5/bias
 : 2Adam/v/conv2d_5/bias
.:, 2Adam/m/conv2d_3/kernel
.:, 2Adam/v/conv2d_3/kernel
 : 2Adam/m/conv2d_3/bias
 : 2Adam/v/conv2d_3/bias
.:, 2"Adam/m/batch_normalization_6/gamma
.:, 2"Adam/v/batch_normalization_6/gamma
-:+ 2!Adam/m/batch_normalization_6/beta
-:+ 2!Adam/v/batch_normalization_6/beta
.:, 2"Adam/m/batch_normalization_4/gamma
.:, 2"Adam/v/batch_normalization_4/gamma
-:+ 2!Adam/m/batch_normalization_4/beta
-:+ 2!Adam/v/batch_normalization_4/beta
.:, 2"Adam/m/batch_normalization_7/gamma
.:, 2"Adam/v/batch_normalization_7/gamma
-:+ 2!Adam/m/batch_normalization_7/beta
-:+ 2!Adam/v/batch_normalization_7/beta
.:, 
2Adam/m/conv2d_6/kernel
.:, 
2Adam/v/conv2d_6/kernel
 :
2Adam/m/conv2d_6/bias
 :
2Adam/v/conv2d_6/bias
ўBЇ
"__inference__update_step_xla_16748gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16753gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16758gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16763gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16768gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16773gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16778gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16783gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16788gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16793gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16798gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16803gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16808gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16813gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16818gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16823gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16828gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16833gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16838gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16843gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16848gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16853gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16858gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16863gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16868gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16873gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16878gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16883gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16888gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
"__inference__update_step_xla_16893gradientvariable"╖
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_conv2d_1_layer_call_fn_1391607inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1391617inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_1_layer_call_fn_1391630inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_1_layer_call_fn_1391643inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391661inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391679inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
тB▀
.__inference_activation_1_layer_call_fn_1391684inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_activation_1_layer_call_and_return_conditional_losses_1391689inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_conv2d_2_layer_call_fn_1391698inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1391708inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▄B┘
(__inference_conv2d_layer_call_fn_1391717inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_conv2d_layer_call_and_return_conditional_losses_1391727inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Б0
В1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_2_layer_call_fn_1391740inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_2_layer_call_fn_1391753inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391771inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391789inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Е0
Ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
·Bў
5__inference_batch_normalization_layer_call_fn_1391802inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
5__inference_batch_normalization_layer_call_fn_1391815inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391833inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391851inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
тB▀
.__inference_activation_2_layer_call_fn_1391856inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_activation_2_layer_call_and_return_conditional_losses_1391861inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_activation_layer_call_fn_1391866inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_activation_layer_call_and_return_conditional_losses_1391871inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
хBт
%__inference_add_layer_call_fn_1391877inputs_0inputs_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
@__inference_add_layer_call_and_return_conditional_losses_1391883inputs_0inputs_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Й0
К1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_3_layer_call_fn_1391896inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_3_layer_call_fn_1391909inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391927inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391945inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
тB▀
.__inference_activation_3_layer_call_fn_1391950inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_activation_3_layer_call_and_return_conditional_losses_1391955inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_conv2d_4_layer_call_fn_1391964inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1391974inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_5_layer_call_fn_1391987inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_5_layer_call_fn_1392000inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392018inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392036inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
тB▀
.__inference_activation_5_layer_call_fn_1392041inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_activation_5_layer_call_and_return_conditional_losses_1392046inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_conv2d_5_layer_call_fn_1392055inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1392065inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_conv2d_3_layer_call_fn_1392074inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1392084inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_6_layer_call_fn_1392097inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_6_layer_call_fn_1392110inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392128inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392146inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_4_layer_call_fn_1392159inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_4_layer_call_fn_1392172inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392190inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392208inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
тB▀
.__inference_activation_6_layer_call_fn_1392213inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_activation_6_layer_call_and_return_conditional_losses_1392218inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
тB▀
.__inference_activation_4_layer_call_fn_1392223inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_activation_4_layer_call_and_return_conditional_losses_1392228inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
чBф
'__inference_add_1_layer_call_fn_1392234inputs_0inputs_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
B__inference_add_1_layer_call_and_return_conditional_losses_1392240inputs_0inputs_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Я0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_7_layer_call_fn_1392253inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_7_layer_call_fn_1392266inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392284inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392302inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
тB▀
.__inference_activation_7_layer_call_fn_1392307inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
I__inference_activation_7_layer_call_and_return_conditional_losses_1392312inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_conv2d_6_layer_call_fn_1392321inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1392332inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
в0
г1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
:  (2total
:  (2count╖
C__inference_ResNet_layer_call_and_return_conditional_losses_1388390яQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_2+                           
p 

 
к "FвC
<К9
tensor_0+                           

Ъ ╖
C__inference_ResNet_layer_call_and_return_conditional_losses_1388511яQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_2+                           
p

 
к "FвC
<К9
tensor_0+                           

Ъ ╢
C__inference_ResNet_layer_call_and_return_conditional_losses_1391301юQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p 

 
к "FвC
<К9
tensor_0+                           

Ъ ╢
C__inference_ResNet_layer_call_and_return_conditional_losses_1391470юQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p

 
к "FвC
<К9
tensor_0+                           

Ъ С
(__inference_ResNet_layer_call_fn_1387725фQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_2+                           
p 

 
к ";К8
unknown+                           
С
(__inference_ResNet_layer_call_fn_1388269фQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_2+                           
p

 
к ";К8
unknown+                           
Р
(__inference_ResNet_layer_call_fn_1391035уQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p 

 
к ";К8
unknown+                           
Р
(__inference_ResNet_layer_call_fn_1391132уQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p

 
к ";К8
unknown+                           
д
"__inference__update_step_xla_16748~xвu
nвk
!К
gradient
<Т9	%в"
·
А
p
` VariableSpec 
`а║─└Л╨?
к "
 М
"__inference__update_step_xla_16753f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`А╣─└Л╨?
к "
 М
"__inference__update_step_xla_16758f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`аИ─└Л╨?
к "
 М
"__inference__update_step_xla_16763f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`└╙╟└Л╨?
к "
 д
"__inference__update_step_xla_16768~xвu
nвk
!К
gradient
<Т9	%в"
·
А
p
` VariableSpec 
`└вМ└Л╨?
к "
 М
"__inference__update_step_xla_16773f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`абМ└Л╨?
к "
 д
"__inference__update_step_xla_16778~xвu
nвk
!К
gradient
<Т9	%в"
·
А
p
` VariableSpec 
`р╪ї╣О╨?
к "
 М
"__inference__update_step_xla_16783f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`рНз╦Л╨?
к "
 М
"__inference__update_step_xla_16788f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`АЇ╟└Л╨?
к "
 М
"__inference__update_step_xla_16793f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`└╗М└Л╨?
к "
 М
"__inference__update_step_xla_16798f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`а·╛└Л╨?
к "
 М
"__inference__update_step_xla_16803f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`рА└└Л╨?
к "
 М
"__inference__update_step_xla_16808f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`└ьУ└Л╨?
к "
 М
"__inference__update_step_xla_16813f`в]
VвS
К
gradient
0Т-	в
·
А
p
` VariableSpec 
`└ёУ└Л╨?
к "
 д
"__inference__update_step_xla_16818~xвu
nвk
!К
gradient 
<Т9	%в"
· 
А
p
` VariableSpec 
`└╙в└Л╨?
к "
 М
"__inference__update_step_xla_16823f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`а╥в└Л╨?
к "
 М
"__inference__update_step_xla_16828f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`аТз└Л╨?
к "
 М
"__inference__update_step_xla_16833f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`аЧз└Л╨?
к "
 д
"__inference__update_step_xla_16838~xвu
nвk
!К
gradient  
<Т9	%в"
·  
А
p
` VariableSpec 
`р╝з└Л╨?
к "
 М
"__inference__update_step_xla_16843f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`└╗з└Л╨?
к "
 д
"__inference__update_step_xla_16848~xвu
nвk
!К
gradient 
<Т9	%в"
· 
А
p
` VariableSpec 
`АеЩ└Л╨?
к "
 М
"__inference__update_step_xla_16853f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`ргЩ└Л╨?
к "
 М
"__inference__update_step_xla_16858f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`р№╬╞Б╨?
к "
 М
"__inference__update_step_xla_16863f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`а├╥╞Б╨?
к "
 М
"__inference__update_step_xla_16868f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`ацЭ└Л╨?
к "
 М
"__inference__update_step_xla_16873f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`аыЭ└Л╨?
к "
 М
"__inference__update_step_xla_16878f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`аН╓╞Б╨?
к "
 М
"__inference__update_step_xla_16883f`в]
VвS
К
gradient 
0Т-	в
· 
А
p
` VariableSpec 
`аТ╓╞Б╨?
к "
 д
"__inference__update_step_xla_16888~xвu
nвk
!К
gradient 

<Т9	%в"
· 

А
p
` VariableSpec 
`└з╓╞Б╨?
к "
 М
"__inference__update_step_xla_16893f`в]
VвS
К
gradient

0Т-	в
·

А
p
` VariableSpec 
`аж╓╞Б╨?
к "
 С
"__inference__wrapped_model_1386853ъQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвJвG
@в=
;К8
input_1+                           
к "IкF
D
ResNet:К7
resnet+                           
с
I__inference_activation_1_layer_call_and_return_conditional_losses_1391689УIвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                           
Ъ ╗
.__inference_activation_1_layer_call_fn_1391684ИIвF
?в<
:К7
inputs+                           
к ";К8
unknown+                           с
I__inference_activation_2_layer_call_and_return_conditional_losses_1391861УIвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                           
Ъ ╗
.__inference_activation_2_layer_call_fn_1391856ИIвF
?в<
:К7
inputs+                           
к ";К8
unknown+                           с
I__inference_activation_3_layer_call_and_return_conditional_losses_1391955УIвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                           
Ъ ╗
.__inference_activation_3_layer_call_fn_1391950ИIвF
?в<
:К7
inputs+                           
к ";К8
unknown+                           с
I__inference_activation_4_layer_call_and_return_conditional_losses_1392228УIвF
?в<
:К7
inputs+                            
к "FвC
<К9
tensor_0+                            
Ъ ╗
.__inference_activation_4_layer_call_fn_1392223ИIвF
?в<
:К7
inputs+                            
к ";К8
unknown+                            с
I__inference_activation_5_layer_call_and_return_conditional_losses_1392046УIвF
?в<
:К7
inputs+                            
к "FвC
<К9
tensor_0+                            
Ъ ╗
.__inference_activation_5_layer_call_fn_1392041ИIвF
?в<
:К7
inputs+                            
к ";К8
unknown+                            с
I__inference_activation_6_layer_call_and_return_conditional_losses_1392218УIвF
?в<
:К7
inputs+                            
к "FвC
<К9
tensor_0+                            
Ъ ╗
.__inference_activation_6_layer_call_fn_1392213ИIвF
?в<
:К7
inputs+                            
к ";К8
unknown+                            с
I__inference_activation_7_layer_call_and_return_conditional_losses_1392312УIвF
?в<
:К7
inputs+                            
к "FвC
<К9
tensor_0+                            
Ъ ╗
.__inference_activation_7_layer_call_fn_1392307ИIвF
?в<
:К7
inputs+                            
к ";К8
unknown+                            ▀
G__inference_activation_layer_call_and_return_conditional_losses_1391871УIвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                           
Ъ ╣
,__inference_activation_layer_call_fn_1391866ИIвF
?в<
:К7
inputs+                           
к ";К8
unknown+                           г
B__inference_add_1_layer_call_and_return_conditional_losses_1392240▄СвН
ЕвБ
Ъ|
<К9
inputs_0+                            
<К9
inputs_1+                            
к "FвC
<К9
tensor_0+                            
Ъ ¤
'__inference_add_1_layer_call_fn_1392234╤СвН
ЕвБ
Ъ|
<К9
inputs_0+                            
<К9
inputs_1+                            
к ";К8
unknown+                            б
@__inference_add_layer_call_and_return_conditional_losses_1391883▄СвН
ЕвБ
Ъ|
<К9
inputs_0+                           
<К9
inputs_1+                           
к "FвC
<К9
tensor_0+                           
Ъ √
%__inference_add_layer_call_fn_1391877╤СвН
ЕвБ
Ъ|
<К9
inputs_0+                           
<К9
inputs_1+                           
к ";К8
unknown+                           Ы
E__inference_add_loss_layer_call_and_return_conditional_losses_1391598Rв
в
К
inputs 
к "0в-
К
tensor_0 
Ъ
К

tensor_1_0 `
*__inference_add_loss_layer_call_fn_13915932в
в
К
inputs 
к "К
unknown Ї
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391661ЭwxyzMвJ
Cв@
:К7
inputs+                           
p 
к "FвC
<К9
tensor_0+                           
Ъ Ї
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391679ЭwxyzMвJ
Cв@
:К7
inputs+                           
p
к "FвC
<К9
tensor_0+                           
Ъ ╬
7__inference_batch_normalization_1_layer_call_fn_1391630ТwxyzMвJ
Cв@
:К7
inputs+                           
p 
к ";К8
unknown+                           ╬
7__inference_batch_normalization_1_layer_call_fn_1391643ТwxyzMвJ
Cв@
:К7
inputs+                           
p
к ";К8
unknown+                           ў
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391771аАБВMвJ
Cв@
:К7
inputs+                           
p 
к "FвC
<К9
tensor_0+                           
Ъ ў
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391789аАБВMвJ
Cв@
:К7
inputs+                           
p
к "FвC
<К9
tensor_0+                           
Ъ ╤
7__inference_batch_normalization_2_layer_call_fn_1391740ХАБВMвJ
Cв@
:К7
inputs+                           
p 
к ";К8
unknown+                           ╤
7__inference_batch_normalization_2_layer_call_fn_1391753ХАБВMвJ
Cв@
:К7
inputs+                           
p
к ";К8
unknown+                           °
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391927бЗИЙКMвJ
Cв@
:К7
inputs+                           
p 
к "FвC
<К9
tensor_0+                           
Ъ °
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1391945бЗИЙКMвJ
Cв@
:К7
inputs+                           
p
к "FвC
<К9
tensor_0+                           
Ъ ╥
7__inference_batch_normalization_3_layer_call_fn_1391896ЦЗИЙКMвJ
Cв@
:К7
inputs+                           
p 
к ";К8
unknown+                           ╥
7__inference_batch_normalization_3_layer_call_fn_1391909ЦЗИЙКMвJ
Cв@
:К7
inputs+                           
p
к ";К8
unknown+                           °
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392190бЩЪЫЬMвJ
Cв@
:К7
inputs+                            
p 
к "FвC
<К9
tensor_0+                            
Ъ °
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1392208бЩЪЫЬMвJ
Cв@
:К7
inputs+                            
p
к "FвC
<К9
tensor_0+                            
Ъ ╥
7__inference_batch_normalization_4_layer_call_fn_1392159ЦЩЪЫЬMвJ
Cв@
:К7
inputs+                            
p 
к ";К8
unknown+                            ╥
7__inference_batch_normalization_4_layer_call_fn_1392172ЦЩЪЫЬMвJ
Cв@
:К7
inputs+                            
p
к ";К8
unknown+                            °
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392018бНОПРMвJ
Cв@
:К7
inputs+                            
p 
к "FвC
<К9
tensor_0+                            
Ъ °
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1392036бНОПРMвJ
Cв@
:К7
inputs+                            
p
к "FвC
<К9
tensor_0+                            
Ъ ╥
7__inference_batch_normalization_5_layer_call_fn_1391987ЦНОПРMвJ
Cв@
:К7
inputs+                            
p 
к ";К8
unknown+                            ╥
7__inference_batch_normalization_5_layer_call_fn_1392000ЦНОПРMвJ
Cв@
:К7
inputs+                            
p
к ";К8
unknown+                            °
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392128бХЦЧШMвJ
Cв@
:К7
inputs+                            
p 
к "FвC
<К9
tensor_0+                            
Ъ °
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1392146бХЦЧШMвJ
Cв@
:К7
inputs+                            
p
к "FвC
<К9
tensor_0+                            
Ъ ╥
7__inference_batch_normalization_6_layer_call_fn_1392097ЦХЦЧШMвJ
Cв@
:К7
inputs+                            
p 
к ";К8
unknown+                            ╥
7__inference_batch_normalization_6_layer_call_fn_1392110ЦХЦЧШMвJ
Cв@
:К7
inputs+                            
p
к ";К8
unknown+                            °
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392284бЭЮЯаMвJ
Cв@
:К7
inputs+                            
p 
к "FвC
<К9
tensor_0+                            
Ъ °
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1392302бЭЮЯаMвJ
Cв@
:К7
inputs+                            
p
к "FвC
<К9
tensor_0+                            
Ъ ╥
7__inference_batch_normalization_7_layer_call_fn_1392253ЦЭЮЯаMвJ
Cв@
:К7
inputs+                            
p 
к ";К8
unknown+                            ╥
7__inference_batch_normalization_7_layer_call_fn_1392266ЦЭЮЯаMвJ
Cв@
:К7
inputs+                            
p
к ";К8
unknown+                            Ў
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391833бГДЕЖMвJ
Cв@
:К7
inputs+                           
p 
к "FвC
<К9
tensor_0+                           
Ъ Ў
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1391851бГДЕЖMвJ
Cв@
:К7
inputs+                           
p
к "FвC
<К9
tensor_0+                           
Ъ ╨
5__inference_batch_normalization_layer_call_fn_1391802ЦГДЕЖMвJ
Cв@
:К7
inputs+                           
p 
к ";К8
unknown+                           ╨
5__inference_batch_normalization_layer_call_fn_1391815ЦГДЕЖMвJ
Cв@
:К7
inputs+                           
p
к ";К8
unknown+                           с
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1391617ЧuvIвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                           
Ъ ╗
*__inference_conv2d_1_layer_call_fn_1391607МuvIвF
?в<
:К7
inputs+                           
к ";К8
unknown+                           с
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1391708Ч{|IвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                           
Ъ ╗
*__inference_conv2d_2_layer_call_fn_1391698М{|IвF
?в<
:К7
inputs+                           
к ";К8
unknown+                           у
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1392084ЩУФIвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                            
Ъ ╜
*__inference_conv2d_3_layer_call_fn_1392074ОУФIвF
?в<
:К7
inputs+                           
к ";К8
unknown+                            у
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1391974ЩЛМIвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                            
Ъ ╜
*__inference_conv2d_4_layer_call_fn_1391964ОЛМIвF
?в<
:К7
inputs+                           
к ";К8
unknown+                            у
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1392065ЩСТIвF
?в<
:К7
inputs+                            
к "FвC
<К9
tensor_0+                            
Ъ ╜
*__inference_conv2d_5_layer_call_fn_1392055ОСТIвF
?в<
:К7
inputs+                            
к ";К8
unknown+                            у
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1392332ЩбвIвF
?в<
:К7
inputs+                            
к "FвC
<К9
tensor_0+                           

Ъ ╜
*__inference_conv2d_6_layer_call_fn_1392321ОбвIвF
?в<
:К7
inputs+                            
к ";К8
unknown+                           
▀
C__inference_conv2d_layer_call_and_return_conditional_losses_1391727Ч}~IвF
?в<
:К7
inputs+                           
к "FвC
<К9
tensor_0+                           
Ъ ╣
(__inference_conv2d_layer_call_fn_1391717М}~IвF
?в<
:К7
inputs+                           
к ";К8
unknown+                           З
___inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_1391587гiвf
_в\
*К'
inputs         ЎЎ

$К!

transforms         
`А
`А
к "6в3
,К)
tensor_0         АА

Ъ с
D__inference_image_projective_transform_layer_1_layer_call_fn_1391579Шiвf
_в\
*К'
inputs         ЎЎ

$К!

transforms         
`А
`А
к "+К(
unknown         АА
Х
]__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_1391573│yвv
oвl
:К7
inputs+                           
$К!

transforms         
`А
`А
к "6в3
,К)
tensor_0         ЎЎ
Ъ я
B__inference_image_projective_transform_layer_layer_call_fn_1391565иyвv
oвl
:К7
inputs+                           
$К!

transforms         
`А
`А
к "+К(
unknown         ЎЎ╦
B__inference_model_layer_call_and_return_conditional_losses_1389620ДQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_1+                           
p 

 
к "[вX
<К9
tensor_0+                           

Ъ
К

tensor_1_0 ╦
B__inference_model_layer_call_and_return_conditional_losses_1389813ДQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_1+                           
p

 
к "[вX
<К9
tensor_0+                           

Ъ
К

tensor_1_0 ╩
B__inference_model_layer_call_and_return_conditional_losses_1390524ГQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p 

 
к "[вX
<К9
tensor_0+                           

Ъ
К

tensor_1_0 ╩
B__inference_model_layer_call_and_return_conditional_losses_1390938ГQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p

 
к "[вX
<К9
tensor_0+                           

Ъ
К

tensor_1_0 Р
'__inference_model_layer_call_fn_1388911фQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_1+                           
p 

 
к ";К8
unknown+                           
Р
'__inference_model_layer_call_fn_1389427фQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвRвO
HвE
;К8
input_1+                           
p

 
к ";К8
unknown+                           
П
'__inference_model_layer_call_fn_1390012уQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p 

 
к ";К8
unknown+                           
П
'__inference_model_layer_call_fn_1390110уQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвQвN
GвD
:К7
inputs+                           
p

 
к ";К8
unknown+                           
И
[__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_1391559иKвH
Aв>
7К4
inp+                           
`А
к "YвV
OвL
$К!

tensor_0_0         
$К!

tensor_0_1         
Ъ ▀
@__inference_random_affine_transform_params_layer_call_fn_1391477ЪKвH
Aв>
7К4
inp+                           
`А
к "KвH
"К
tensor_0         
"К
tensor_1         Я
%__inference_signature_wrapper_1389914їQuvwxyz}~{|ГДЕЖАБВЗИЙКЛМНОПРУФСТЩЪЫЬХЦЧШЭЮЯабвUвR
в 
KкH
F
input_1;К8
input_1+                           "IкF
D
ResNet:К7
resnet+                           
