єО7
п#√#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
X
AdjustContrastv2
images"T
contrast_factor
output"T"
Ttype0:
2
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
ы
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
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
—
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЌћL>"
Ttype0:
2
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
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
 И"serve*2.9.32v2.9.2-107-ga5ed5f39b678ъћ0
А
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0
Р
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/v
Й
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/v
У
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/v
Х
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/v
У
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_4/gamma/v
Х
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_6/beta/v
У
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_6/gamma/v
Х
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
:@*
dtype0
А
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
С
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*'
shared_nameAdam/conv2d_3/kernel/v
К
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*'
_output_shapes
:А@*
dtype0
А
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_5/kernel/v
Й
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_5/beta/v
У
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_5/gamma/v
Х
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:@*
dtype0
А
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:@*
dtype0
С
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*'
shared_nameAdam/conv2d_4/kernel/v
К
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*'
_output_shapes
:А@*
dtype0
Ы
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/v
Ф
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/v
Ц
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:А*
dtype0
Ч
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/batch_normalization/beta/v
Р
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:А*
dtype0
Щ
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/batch_normalization/gamma/v
Т
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_2/beta/v
Ф
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_2/gamma/v
Ц
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:А*
dtype0
}
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_nameAdam/conv2d/bias/v
v
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes	
:А*
dtype0
Н
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d/kernel/v
Ж
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*'
_output_shapes
:А*
dtype0
Б
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_2/bias/v
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_2/kernel/v
Л
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*(
_output_shapes
:АА*
dtype0
Ы
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_1/beta/v
Ф
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_1/gamma/v
Ц
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:А*
dtype0
Б
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_1/bias/v
z
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes	
:А*
dtype0
С
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/conv2d_1/kernel/v
К
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*'
_output_shapes
:А*
dtype0
А
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/m
Й
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_7/beta/m
У
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_7/gamma/m
Х
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/m
У
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_4/gamma/m
Х
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_6/beta/m
У
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_6/gamma/m
Х
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
:@*
dtype0
А
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
С
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*'
shared_nameAdam/conv2d_3/kernel/m
К
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*'
_output_shapes
:А@*
dtype0
А
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_5/kernel/m
Й
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:@@*
dtype0
Ъ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_5/beta/m
У
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:@*
dtype0
Ь
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_5/gamma/m
Х
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:@*
dtype0
А
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:@*
dtype0
С
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*'
shared_nameAdam/conv2d_4/kernel/m
К
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*'
_output_shapes
:А@*
dtype0
Ы
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/m
Ф
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/m
Ц
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:А*
dtype0
Ч
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adam/batch_normalization/beta/m
Р
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:А*
dtype0
Щ
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/batch_normalization/gamma/m
Т
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_2/beta/m
Ф
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_2/gamma/m
Ц
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:А*
dtype0
}
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*#
shared_nameAdam/conv2d/bias/m
v
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes	
:А*
dtype0
Н
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d/kernel/m
Ж
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*'
_output_shapes
:А*
dtype0
Б
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_2/bias/m
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_2/kernel/m
Л
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*(
_output_shapes
:АА*
dtype0
Ы
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_1/beta/m
Ф
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_1/gamma/m
Ц
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:А*
dtype0
Б
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_1/bias/m
z
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes	
:А*
dtype0
С
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/conv2d_1/kernel/m
К
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*'
_output_shapes
:А*
dtype0
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
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
Г
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:А@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
Г
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@* 
shared_nameconv2d_4/kernel
|
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:А@*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_3/moving_variance
Ь
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_3/moving_mean
Ф
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:А*
dtype0
Н
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_3/beta
Ж
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:А*
dtype0
П
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_3/gamma
И
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
Я
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#batch_normalization/moving_variance
Ш
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:А*
dtype0
Ч
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!batch_normalization/moving_mean
Р
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:А*
dtype0
Й
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_namebatch_normalization/beta
В
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:А*
dtype0
Л
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namebatch_normalization/gamma
Д
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:А*
dtype0
£
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_2/moving_variance
Ь
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_2/moving_mean
Ф
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:А*
dtype0
Н
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_2/beta
Ж
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:А*
dtype0
П
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_2/gamma
И
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:А*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:А*
dtype0

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:А*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:АА*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_1/moving_variance
Ь
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_1/moving_mean
Ф
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:А*
dtype0
Н
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_1/beta
Ж
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:А*
dtype0
П
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_1/gamma
И
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:А*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:А*
dtype0
Г
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:А*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *У:Г?

NoOpNoOp
ҐТ
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*ЏС
valueѕСBЋС B√С
ь
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
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&loss
'
signatures*
* 
э
(layer-0
)layer_with_weights-0
)layer-1
*layer_with_weights-1
*layer-2
+layer-3
,layer_with_weights-2
,layer-4
-layer_with_weights-3
-layer-5
.layer_with_weights-4
.layer-6
/layer_with_weights-5
/layer-7
0layer-8
1layer-9
2layer-10
3layer_with_weights-6
3layer-11
4layer-12
5layer_with_weights-7
5layer-13
6layer_with_weights-8
6layer-14
7layer-15
8layer_with_weights-9
8layer-16
9layer_with_weights-10
9layer-17
:layer_with_weights-11
:layer-18
;layer_with_weights-12
;layer-19
<layer-20
=layer-21
>layer-22
?layer_with_weights-13
?layer-23
@layer-24
Alayer_with_weights-14
Alayer-25
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
О
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
О
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses* 

T	keras_api* 

U	keras_api* 
О
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
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

o	keras_api* 

p	keras_api* 
О
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
П
w0
x1
y2
z3
{4
|5
}6
~7
8
А9
Б10
В11
Г12
Д13
Е14
Ж15
З16
И17
Й18
К19
Л20
М21
Н22
О23
П24
Р25
С26
Т27
У28
Ф29
Х30
Ц31
Ч32
Ш33
Щ34
Ъ35
Ы36
Ь37
Э38
Ю39
Я40
†41
°42
Ґ43
£44
§45*
Б
w0
x1
y2
z3
}4
~5
6
А7
Б8
В9
Е10
Ж11
Й12
К13
Н14
О15
П16
Р17
У18
Ф19
Х20
Ц21
Ч22
Ш23
Ы24
Ь25
Я26
†27
£28
§29*
* 
µ
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
:
™trace_0
Ђtrace_1
ђtrace_2
≠trace_3* 
:
Ѓtrace_0
ѓtrace_1
∞trace_2
±trace_3* 
* 
ї
	≤iter
≥beta_1
іbeta_2

µdecaywm…xm ymЋzmћ}mЌ~mќmѕ	Аm–	Бm—	Вm“	Еm”	Жm‘	Йm’	Кm÷	Нm„	ОmЎ	Пmў	РmЏ	Уmџ	Фm№	ХmЁ	Цmё	Чmя	Шmа	Ыmб	Ьmв	Яmг	†mд	£mе	§mжwvзxvиyvйzvк}vл~vмvн	Аvо	Бvп	Вvр	Еvс	Жvт	Йvу	Кvф	Нvх	Оvц	Пvч	Рvш	Уvщ	Фvъ	Хvы	Цvь	Чvэ	Шvю	Ыv€	ЬvА	ЯvБ	†vВ	£vГ	§vД*
* 

ґserving_default* 
* 
ѕ
Ј	variables
Єtrainable_variables
єregularization_losses
Ї	keras_api
ї__call__
+Љ&call_and_return_all_conditional_losses

wkernel
xbias
!љ_jit_compiled_convolution_op*
№
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
¬__call__
+√&call_and_return_all_conditional_losses
	ƒaxis
	ygamma
zbeta
{moving_mean
|moving_variance*
Ф
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses* 
ѕ
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses

}kernel
~bias
!—_jit_compiled_convolution_op*
–
“	variables
”trainable_variables
‘regularization_losses
’	keras_api
÷__call__
+„&call_and_return_all_conditional_losses

kernel
	Аbias
!Ў_jit_compiled_convolution_op*
а
ў	variables
Џtrainable_variables
џregularization_losses
№	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses
	яaxis

Бgamma
	Вbeta
Гmoving_mean
Дmoving_variance*
а
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
	жaxis

Еgamma
	Жbeta
Зmoving_mean
Иmoving_variance*
Ф
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses* 
Ф
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses* 
Ф
у	variables
фtrainable_variables
хregularization_losses
ц	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses* 
а
щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses
	€axis

Йgamma
	Кbeta
Лmoving_mean
Мmoving_variance*
Ф
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses* 
—
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses
Нkernel
	Оbias
!М_jit_compiled_convolution_op*
а
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
	Уaxis

Пgamma
	Рbeta
Сmoving_mean
Тmoving_variance*
Ф
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses* 
—
Ъ	variables
Ыtrainable_variables
Ьregularization_losses
Э	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses
Уkernel
	Фbias
!†_jit_compiled_convolution_op*
—
°	variables
Ґtrainable_variables
£regularization_losses
§	keras_api
•__call__
+¶&call_and_return_all_conditional_losses
Хkernel
	Цbias
!І_jit_compiled_convolution_op*
а
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses
	Ѓaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance*
а
ѓ	variables
∞trainable_variables
±regularization_losses
≤	keras_api
≥__call__
+і&call_and_return_all_conditional_losses
	µaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance*
Ф
ґ	variables
Јtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses* 
Ф
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses* 
Ф
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
∆__call__
+«&call_and_return_all_conditional_losses* 
а
»	variables
…trainable_variables
 regularization_losses
Ћ	keras_api
ћ__call__
+Ќ&call_and_return_all_conditional_losses
	ќaxis

Яgamma
	†beta
°moving_mean
Ґmoving_variance*
Ф
ѕ	variables
–trainable_variables
—regularization_losses
“	keras_api
”__call__
+‘&call_and_return_all_conditional_losses* 
—
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses
£kernel
	§bias
!џ_jit_compiled_convolution_op*
П
w0
x1
y2
z3
{4
|5
}6
~7
8
А9
Б10
В11
Г12
Д13
Е14
Ж15
З16
И17
Й18
К19
Л20
М21
Н22
О23
П24
Р25
С26
Т27
У28
Ф29
Х30
Ц31
Ч32
Ш33
Щ34
Ъ35
Ы36
Ь37
Э38
Ю39
Я40
†41
°42
Ґ43
£44
§45*
Б
w0
x1
y2
z3
}4
~5
6
А7
Б8
В9
Е10
Ж11
Й12
К13
Н14
О15
П16
Р17
У18
Ф19
Х20
Ц21
Ч22
Ш23
Ы24
Ь25
Я26
†27
£28
§29*
* 
Ш
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
:
бtrace_0
вtrace_1
гtrace_2
дtrace_3* 
:
еtrace_0
жtrace_1
зtrace_2
иtrace_3* 
* 
* 
* 
Ц
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

оtrace_0* 

пtrace_0* 
* 
* 
* 
Ц
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

хtrace_0* 

цtrace_0* 
* 
* 
* 
* 
* 
Ц
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

ьtrace_0* 

эtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
юnon_trainable_variables
€layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

Гtrace_0* 

Дtrace_0* 
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
{0
|1
Г2
Д3
З4
И5
Л6
М7
С8
Т9
Щ10
Ъ11
Э12
Ю13
°14
Ґ15*
в
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
28*

Е0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 

w0
x1*

w0
x1*
* 
Ю
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
Ј	variables
Єtrainable_variables
єregularization_losses
ї__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses*

Лtrace_0* 

Мtrace_0* 
* 
 
y0
z1
{2
|3*

y0
z1*
* 
Ю
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Њ	variables
њtrainable_variables
јregularization_losses
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses*

Тtrace_0
Уtrace_1* 

Фtrace_0
Хtrace_1* 
* 
* 
* 
* 
Ь
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 

Ыtrace_0* 

Ьtrace_0* 

}0
~1*

}0
~1*
* 
Ю
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses*

Ґtrace_0* 

£trace_0* 
* 

0
А1*

0
А1*
* 
Ю
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
“	variables
”trainable_variables
‘regularization_losses
÷__call__
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses*

©trace_0* 

™trace_0* 
* 
$
Б0
В1
Г2
Д3*

Б0
В1*
* 
Ю
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
ў	variables
Џtrainable_variables
џregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses*

∞trace_0
±trace_1* 

≤trace_0
≥trace_1* 
* 
$
Е0
Ж1
З2
И3*

Е0
Ж1*
* 
Ю
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses*

єtrace_0
Їtrace_1* 

їtrace_0
Љtrace_1* 
* 
* 
* 
* 
Ь
љnon_trainable_variables
Њlayers
њmetrics
 јlayer_regularization_losses
Ѕlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses* 

¬trace_0* 

√trace_0* 
* 
* 
* 
Ь
ƒnon_trainable_variables
≈layers
∆metrics
 «layer_regularization_losses
»layer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses* 

…trace_0* 

 trace_0* 
* 
* 
* 
Ь
Ћnon_trainable_variables
ћlayers
Ќmetrics
 ќlayer_regularization_losses
ѕlayer_metrics
у	variables
фtrainable_variables
хregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses* 

–trace_0* 

—trace_0* 
$
Й0
К1
Л2
М3*

Й0
К1*
* 
Ю
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses*

„trace_0
Ўtrace_1* 

ўtrace_0
Џtrace_1* 
* 
* 
* 
* 
Ь
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses* 

аtrace_0* 

бtrace_0* 

Н0
О1*

Н0
О1*
* 
Ю
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses*

зtrace_0* 

иtrace_0* 
* 
$
П0
Р1
С2
Т3*

П0
Р1*
* 
Ю
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses*

оtrace_0
пtrace_1* 

рtrace_0
сtrace_1* 
* 
* 
* 
* 
Ь
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses* 

чtrace_0* 

шtrace_0* 

У0
Ф1*

У0
Ф1*
* 
Ю
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
Ъ	variables
Ыtrainable_variables
Ьregularization_losses
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses*

юtrace_0* 

€trace_0* 
* 

Х0
Ц1*

Х0
Ц1*
* 
Ю
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
°	variables
Ґtrainable_variables
£regularization_losses
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
* 
$
Ч0
Ш1
Щ2
Ъ3*

Ч0
Ш1*
* 
Ю
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses*

Мtrace_0
Нtrace_1* 

Оtrace_0
Пtrace_1* 
* 
$
Ы0
Ь1
Э2
Ю3*

Ы0
Ь1*
* 
Ю
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
ѓ	variables
∞trainable_variables
±regularization_losses
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses*

Хtrace_0
Цtrace_1* 

Чtrace_0
Шtrace_1* 
* 
* 
* 
* 
Ь
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
ґ	variables
Јtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses* 

Юtrace_0* 

Яtrace_0* 
* 
* 
* 
Ь
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses* 

•trace_0* 

¶trace_0* 
* 
* 
* 
Ь
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
¬	variables
√trainable_variables
ƒregularization_losses
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses* 

ђtrace_0* 

≠trace_0* 
$
Я0
†1
°2
Ґ3*

Я0
†1*
* 
Ю
Ѓnon_trainable_variables
ѓlayers
∞metrics
 ±layer_regularization_losses
≤layer_metrics
»	variables
…trainable_variables
 regularization_losses
ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses*

≥trace_0
іtrace_1* 

µtrace_0
ґtrace_1* 
* 
* 
* 
* 
Ь
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
ѕ	variables
–trainable_variables
—regularization_losses
”__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses* 

Љtrace_0* 

љtrace_0* 

£0
§1*

£0
§1*
* 
Ю
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
’	variables
÷trainable_variables
„regularization_losses
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses*

√trace_0* 

ƒtrace_0* 
* 
И
{0
|1
Г2
Д3
З4
И5
Л6
М7
С8
Т9
Щ10
Ъ11
Э12
Ю13
°14
Ґ15*
 
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21
>22
?23
@24
A25*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
≈	variables
∆	keras_api

«total

»count*
* 
* 
* 
* 
* 
* 
* 

{0
|1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
Г0
Д1*
* 
* 
* 
* 
* 
* 
* 
* 

З0
И1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
Л0
М1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
С0
Т1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
Щ0
Ъ1*
* 
* 
* 
* 
* 
* 
* 
* 

Э0
Ю1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
°0
Ґ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
«0
»1*

≈	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/batch_normalization/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/batch_normalization/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_4/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_4/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_5/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_5/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_3/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_3/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_6/kernel/mCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_6/bias/mCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_2/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/batch_normalization/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/batch_normalization/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_4/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_4/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_5/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_5/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_3/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_3/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_6/kernel/vCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_6/bias/vCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ
serving_default_input_1Placeholder*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
†
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d/kernelconv2d/biasconv2d_2/kernelconv2d_2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_3/kernelconv2d_3/biasconv2d_5/kernelconv2d_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_6/kernelconv2d_6/biasConst*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *,
f'R%
#__inference_signature_wrapper_34392
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
а,
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOpConst_1*}
Tinv
t2r	*
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
GPU2 *0J 8В *'
f"R 
__inference__traced_save_37180
Щ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasconv2d/kernelconv2d/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_5/kernelconv2d_5/biasconv2d_3/kernelconv2d_3/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_6/kernelconv2d_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcountAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/v*|
Tinu
s2q*
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
GPU2 *0J 8В **
f%R#
!__inference__traced_restore_37526ул+
Д	
Д
>__inference_random_affine_transform_params_layer_call_fn_35965
inp
identity

identity_1ИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinp*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *b
f]R[
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_33149o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:f b
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

_user_specified_nameinp
Ћ
Ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31777

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ЛE
Е
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_33149
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
valueB:—
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
:€€€€€€€€€*
dtype0c
RoundRound%random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:€€€€€€€€€J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @S
mulMul	Round:y:0mul/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Q
subSubmul:z:0sub/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€d
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
 *џIјY
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џI@Л
random_uniform_1/RandomUniformRandomUniformrandom_uniform_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: М
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€А
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€N
CosCosrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€N
SinSinrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€A
NegNegSin:y:0*
T0*#
_output_shapes
:€€€€€€€€€L
mul_1MulNeg:y:0sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
Sin_1Sinrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
Cos_1Cosrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€N
mul_2Mul	Cos_1:y:0sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€_
packed/0PackCos:y:0	mul_1:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€a
packed/1Pack	Sin_1:y:0	mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€s
packedPackpacked/0:output:0packed/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€a

packed_1/0PackCos:y:0	Sin_1:y:0*
N*
T0*'
_output_shapes
:€€€€€€€€€c

packed_1/1Pack	mul_1:z:0	mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€y
packed_1Packpacked_1/0:output:0packed_1/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_1	Transposepacked_1:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
ConstConst*
_output_shapes

:*
dtype0*!
valueB"у5Cу5C`
Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   Cn
MatMulBatchMatMulV2transpose:y:0Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€c
sub_1SubConst:output:0MatMul:output:0*
T0*+
_output_shapes
:€€€€€€€€€k
MatMul_1BatchMatMulV2transpose_1:y:0	sub_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
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
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskT
Neg_1Negstrided_slice_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€j
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
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskT
Neg_2Negstrided_slice_2:output:0*
T0*#
_output_shapes
:€€€€€€€€€E
Neg_3Neg	Neg_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€N
mul_3Mul	Neg_3:y:0Cos:y:0*
T0*#
_output_shapes
:€€€€€€€€€P
mul_4Mul	Neg_2:y:0	mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
sub_2Sub	mul_3:z:0	mul_4:z:0*
T0*#
_output_shapes
:€€€€€€€€€E
Neg_4Neg	Neg_2:y:0*
T0*#
_output_shapes
:€€€€€€€€€P
mul_5Mul	Neg_4:y:0	mul_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
mul_6Mul	Neg_1:y:0	Sin_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€P
sub_3Sub	mul_5:z:0	mul_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€f
zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€s
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
:€€€€€€€€€h
zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
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
:€€€€€€€€€ґ
stackPackCos:y:0	mul_1:z:0	sub_2:z:0	Sin_1:y:0	mul_2:z:0	sub_3:z:0zeros:output:0zeros_1:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

axish
zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
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
:€€€€€€€€€h
zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
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
:€€€€€€€€€Ї
stack_1PackCos:y:0	Sin_1:y:0	Neg_1:y:0	mul_1:z:0	mul_2:z:0	Neg_2:y:0zeros_2:output:0zeros_3:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

axisX
IdentityIdentitystack_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€X

Identity_1Identitystack:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:f b
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

_user_specified_nameinp
ъ
ґ
A__inference_ResNet_layer_call_and_return_conditional_losses_32531

inputs)
conv2d_1_32413:А
conv2d_1_32415:	А*
batch_normalization_1_32418:	А*
batch_normalization_1_32420:	А*
batch_normalization_1_32422:	А*
batch_normalization_1_32424:	А'
conv2d_32428:А
conv2d_32430:	А*
conv2d_2_32433:АА
conv2d_2_32435:	А(
batch_normalization_32438:	А(
batch_normalization_32440:	А(
batch_normalization_32442:	А(
batch_normalization_32444:	А*
batch_normalization_2_32447:	А*
batch_normalization_2_32449:	А*
batch_normalization_2_32451:	А*
batch_normalization_2_32453:	А*
batch_normalization_3_32459:	А*
batch_normalization_3_32461:	А*
batch_normalization_3_32463:	А*
batch_normalization_3_32465:	А)
conv2d_4_32469:А@
conv2d_4_32471:@)
batch_normalization_5_32474:@)
batch_normalization_5_32476:@)
batch_normalization_5_32478:@)
batch_normalization_5_32480:@)
conv2d_3_32484:А@
conv2d_3_32486:@(
conv2d_5_32489:@@
conv2d_5_32491:@)
batch_normalization_4_32494:@)
batch_normalization_4_32496:@)
batch_normalization_4_32498:@)
batch_normalization_4_32500:@)
batch_normalization_6_32503:@)
batch_normalization_6_32505:@)
batch_normalization_6_32507:@)
batch_normalization_6_32509:@)
batch_normalization_7_32515:@)
batch_normalization_7_32517:@)
batch_normalization_7_32519:@)
batch_normalization_7_32521:@(
conv2d_6_32525:@
conv2d_6_32527:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallН
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_32413conv2d_1_32415*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_31836†
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_32418batch_normalization_1_32420batch_normalization_1_32422batch_normalization_1_32424*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31360С
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_31856Е
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32428conv2d_32430*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_31868≠
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv2d_2_32433conv2d_2_32435*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_31884Т
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_32438batch_normalization_32440batch_normalization_32442batch_normalization_32444*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31488†
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_32447batch_normalization_2_32449batch_normalization_2_32451batch_normalization_2_32453*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31424С
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_31913Л
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_31920Ф
add/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_31928У
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_32459batch_normalization_3_32461batch_normalization_3_32463batch_normalization_3_32465*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31552С
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_31944ђ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_4_32469conv2d_4_32471*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_31956Я
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_32474batch_normalization_5_32476batch_normalization_5_32478batch_normalization_5_32480*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31616Р
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_31976ђ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_3_32484conv2d_3_32486*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_31988ђ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0conv2d_5_32489conv2d_5_32491*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_32004Я
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_32494batch_normalization_4_32496batch_normalization_4_32498batch_normalization_4_32500*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31744Я
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_32503batch_normalization_6_32505batch_normalization_6_32507batch_normalization_6_32509*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31680Р
leaky_re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_32033Р
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_32040Щ
add_1/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_32048Ф
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_32515batch_normalization_7_32517batch_normalization_7_32519batch_normalization_7_32521*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31808Р
leaky_re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_32064ђ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv2d_6_32525conv2d_6_32527*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_32077Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ќ
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_36443

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Т
G
+__inference_leaky_re_lu_layer_call_fn_36354

inputs
identity—
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_31920{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
џf
†
@__inference_model_layer_call_and_return_conditional_losses_34091
input_1'
resnet_33898:А
resnet_33900:	А
resnet_33902:	А
resnet_33904:	А
resnet_33906:	А
resnet_33908:	А'
resnet_33910:А
resnet_33912:	А(
resnet_33914:АА
resnet_33916:	А
resnet_33918:	А
resnet_33920:	А
resnet_33922:	А
resnet_33924:	А
resnet_33926:	А
resnet_33928:	А
resnet_33930:	А
resnet_33932:	А
resnet_33934:	А
resnet_33936:	А
resnet_33938:	А
resnet_33940:	А'
resnet_33942:А@
resnet_33944:@
resnet_33946:@
resnet_33948:@
resnet_33950:@
resnet_33952:@'
resnet_33954:А@
resnet_33956:@&
resnet_33958:@@
resnet_33960:@
resnet_33962:@
resnet_33964:@
resnet_33966:@
resnet_33968:@
resnet_33970:@
resnet_33972:@
resnet_33974:@
resnet_33976:@
resnet_33978:@
resnet_33980:@
resnet_33982:@
resnet_33984:@&
resnet_33986:@
resnet_33988:<
8tf_image_adjust_contrast_adjust_contrast_contrast_factor
identity

identity_1ИҐResNet/StatefulPartitionedCallҐ ResNet/StatefulPartitionedCall_1Ґ6random_affine_transform_params/StatefulPartitionedCall≈
ResNet/StatefulPartitionedCallStatefulPartitionedCallinput_1resnet_33898resnet_33900resnet_33902resnet_33904resnet_33906resnet_33908resnet_33910resnet_33912resnet_33914resnet_33916resnet_33918resnet_33920resnet_33922resnet_33924resnet_33926resnet_33928resnet_33930resnet_33932resnet_33934resnet_33936resnet_33938resnet_33940resnet_33942resnet_33944resnet_33946resnet_33948resnet_33950resnet_33952resnet_33954resnet_33956resnet_33958resnet_33960resnet_33962resnet_33964resnet_33966resnet_33968resnet_33970resnet_33972resnet_33974resnet_33976resnet_33978resnet_33980resnet_33982resnet_33984resnet_33986resnet_33988*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32084~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             “
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *b
f]R[
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_33149є
0image_projective_transform_layer/PartitionedCallPartitionedCallinput_1?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_33160О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             µ
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€д
(tf.image.adjust_contrast/adjust_contrastAdjustContrastv29image_projective_transform_layer/PartitionedCall:output:08tf_image_adjust_contrast_adjust_contrast_contrast_factor*1
_output_shapes
:€€€€€€€€€ццђ
1tf.image.adjust_contrast/adjust_contrast/IdentityIdentity1tf.image.adjust_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццк
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall:tf.image.adjust_contrast/adjust_contrast/Identity:output:0resnet_33898resnet_33900resnet_33902resnet_33904resnet_33906resnet_33908resnet_33910resnet_33912resnet_33914resnet_33916resnet_33918resnet_33920resnet_33922resnet_33924resnet_33926resnet_33928resnet_33930resnet_33932resnet_33934resnet_33936resnet_33938resnet_33940resnet_33942resnet_33944resnet_33946resnet_33948resnet_33950resnet_33952resnet_33954resnet_33956resnet_33958resnet_33960resnet_33962resnet_33964resnet_33966resnet_33968resnet_33970resnet_33972resnet_33974resnet_33976resnet_33978resnet_33980resnet_33982resnet_33984resnet_33986resnet_33988*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32084я
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_33222А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Џ
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА€€€€€€€€€Џ
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
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
valueB"            ч
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А@Ц
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ©
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€©
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€≠
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"€€€€ю€€€С
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
value	B :≥
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: …
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
GPU2 *0J 8В *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_33267Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: √
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:/

_output_shapes
: 
Х
√
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36433

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Е
њ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31616

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ъ	
‘
5__inference_batch_normalization_1_layer_call_fn_36131

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31360К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ц
I
-__inference_leaky_re_lu_2_layer_call_fn_36344

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_31913{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Т
I
-__inference_leaky_re_lu_4_layer_call_fn_36711

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_32040z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Х
√
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31424

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
 
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_36800

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Т	
–
5__inference_batch_normalization_4_layer_call_fn_36660

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31744Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ъ	
‘
5__inference_batch_normalization_3_layer_call_fn_36397

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31552К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Б
n
B__inference_image_projective_transform_layer_1_layer_call_fn_36067

inputs

transforms
identityд
PartitionedCallPartitionedCallinputs
transforms*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_33222j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€цц:€€€€€€€€€:Y U
1
_output_shapes
:€€€€€€€€€цц
 
_user_specified_nameinputs:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
transforms
Х
√
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31552

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ф
ь
C__inference_conv2d_6_layer_call_and_return_conditional_losses_32077

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€p
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€z
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Кз
ЂA
@__inference_model_layer_call_and_return_conditional_losses_35009

inputsI
.resnet_conv2d_1_conv2d_readvariableop_resource:А>
/resnet_conv2d_1_biasadd_readvariableop_resource:	АC
4resnet_batch_normalization_1_readvariableop_resource:	АE
6resnet_batch_normalization_1_readvariableop_1_resource:	АT
Eresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АV
Gresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	АG
,resnet_conv2d_conv2d_readvariableop_resource:А<
-resnet_conv2d_biasadd_readvariableop_resource:	АJ
.resnet_conv2d_2_conv2d_readvariableop_resource:АА>
/resnet_conv2d_2_biasadd_readvariableop_resource:	АA
2resnet_batch_normalization_readvariableop_resource:	АC
4resnet_batch_normalization_readvariableop_1_resource:	АR
Cresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:	АT
Eresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	АC
4resnet_batch_normalization_2_readvariableop_resource:	АE
6resnet_batch_normalization_2_readvariableop_1_resource:	АT
Eresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АV
Gresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	АC
4resnet_batch_normalization_3_readvariableop_resource:	АE
6resnet_batch_normalization_3_readvariableop_1_resource:	АT
Eresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АV
Gresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АI
.resnet_conv2d_4_conv2d_readvariableop_resource:А@=
/resnet_conv2d_4_biasadd_readvariableop_resource:@B
4resnet_batch_normalization_5_readvariableop_resource:@D
6resnet_batch_normalization_5_readvariableop_1_resource:@S
Eresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@I
.resnet_conv2d_3_conv2d_readvariableop_resource:А@=
/resnet_conv2d_3_biasadd_readvariableop_resource:@H
.resnet_conv2d_5_conv2d_readvariableop_resource:@@=
/resnet_conv2d_5_biasadd_readvariableop_resource:@B
4resnet_batch_normalization_4_readvariableop_resource:@D
6resnet_batch_normalization_4_readvariableop_1_resource:@S
Eresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@B
4resnet_batch_normalization_6_readvariableop_resource:@D
6resnet_batch_normalization_6_readvariableop_1_resource:@S
Eresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@B
4resnet_batch_normalization_7_readvariableop_resource:@D
6resnet_batch_normalization_7_readvariableop_1_resource:@S
Eresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@H
.resnet_conv2d_6_conv2d_readvariableop_resource:@=
/resnet_conv2d_6_biasadd_readvariableop_resource:<
8tf_image_adjust_contrast_adjust_contrast_contrast_factor
identity

identity_1ИҐ:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpҐ>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Ґ)ResNet/batch_normalization/ReadVariableOpҐ+ResNet/batch_normalization/ReadVariableOp_1Ґ+ResNet/batch_normalization/ReadVariableOp_2Ґ+ResNet/batch_normalization/ReadVariableOp_3Ґ<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_1/ReadVariableOpҐ-ResNet/batch_normalization_1/ReadVariableOp_1Ґ-ResNet/batch_normalization_1/ReadVariableOp_2Ґ-ResNet/batch_normalization_1/ReadVariableOp_3Ґ<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_2/ReadVariableOpҐ-ResNet/batch_normalization_2/ReadVariableOp_1Ґ-ResNet/batch_normalization_2/ReadVariableOp_2Ґ-ResNet/batch_normalization_2/ReadVariableOp_3Ґ<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_3/ReadVariableOpҐ-ResNet/batch_normalization_3/ReadVariableOp_1Ґ-ResNet/batch_normalization_3/ReadVariableOp_2Ґ-ResNet/batch_normalization_3/ReadVariableOp_3Ґ<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_4/ReadVariableOpҐ-ResNet/batch_normalization_4/ReadVariableOp_1Ґ-ResNet/batch_normalization_4/ReadVariableOp_2Ґ-ResNet/batch_normalization_4/ReadVariableOp_3Ґ<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_5/ReadVariableOpҐ-ResNet/batch_normalization_5/ReadVariableOp_1Ґ-ResNet/batch_normalization_5/ReadVariableOp_2Ґ-ResNet/batch_normalization_5/ReadVariableOp_3Ґ<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_6/ReadVariableOpҐ-ResNet/batch_normalization_6/ReadVariableOp_1Ґ-ResNet/batch_normalization_6/ReadVariableOp_2Ґ-ResNet/batch_normalization_6/ReadVariableOp_3Ґ<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_7/ReadVariableOpҐ-ResNet/batch_normalization_7/ReadVariableOp_1Ґ-ResNet/batch_normalization_7/ReadVariableOp_2Ґ-ResNet/batch_normalization_7/ReadVariableOp_3Ґ$ResNet/conv2d/BiasAdd/ReadVariableOpҐ&ResNet/conv2d/BiasAdd_1/ReadVariableOpҐ#ResNet/conv2d/Conv2D/ReadVariableOpҐ%ResNet/conv2d/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_1/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_1/Conv2D/ReadVariableOpҐ'ResNet/conv2d_1/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_2/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_2/Conv2D/ReadVariableOpҐ'ResNet/conv2d_2/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_3/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_3/Conv2D/ReadVariableOpҐ'ResNet/conv2d_3/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_4/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_4/Conv2D/ReadVariableOpҐ'ResNet/conv2d_4/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_5/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_5/Conv2D/ReadVariableOpҐ'ResNet/conv2d_5/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_6/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_6/Conv2D/ReadVariableOpҐ'ResNet/conv2d_6/Conv2D_1/ReadVariableOpЭ
%ResNet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ћ
ResNet/conv2d_1/Conv2DConv2Dinputs-ResNet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
У
&ResNet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ј
ResNet/conv2d_1/BiasAddBiasAddResNet/conv2d_1/Conv2D:output:0.ResNet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЭ
+ResNet/batch_normalization_1/ReadVariableOpReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_1/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0√
>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ш
-ResNet/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_1/BiasAdd:output:03ResNet/batch_normalization_1/ReadVariableOp:value:05ResNet/batch_normalization_1/ReadVariableOp_1:value:0DResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ≤
ResNet/leaky_re_lu_1/LeakyRelu	LeakyRelu1ResNet/batch_normalization_1/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Щ
#ResNet/conv2d/Conv2D/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0»
ResNet/conv2d/Conv2DConv2Dinputs+ResNet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
П
$ResNet/conv2d/BiasAdd/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
ResNet/conv2d/BiasAddBiasAddResNet/conv2d/Conv2D:output:0,ResNet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЮ
%ResNet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0т
ResNet/conv2d_2/Conv2DConv2D,ResNet/leaky_re_lu_1/LeakyRelu:activations:0-ResNet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
У
&ResNet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ј
ResNet/conv2d_2/BiasAddBiasAddResNet/conv2d_2/Conv2D:output:0.ResNet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЩ
)ResNet/batch_normalization/ReadVariableOpReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0Э
+ResNet/batch_normalization/ReadVariableOp_1ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ї
:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0м
+ResNet/batch_normalization/FusedBatchNormV3FusedBatchNormV3ResNet/conv2d/BiasAdd:output:01ResNet/batch_normalization/ReadVariableOp:value:03ResNet/batch_normalization/ReadVariableOp_1:value:0BResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0DResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( Э
+ResNet/batch_normalization_2/ReadVariableOpReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_2/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0√
>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ш
-ResNet/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_2/BiasAdd:output:03ResNet/batch_normalization_2/ReadVariableOp:value:05ResNet/batch_normalization_2/ReadVariableOp_1:value:0DResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ≤
ResNet/leaky_re_lu_2/LeakyRelu	LeakyRelu1ResNet/batch_normalization_2/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Ѓ
ResNet/leaky_re_lu/LeakyRelu	LeakyRelu/ResNet/batch_normalization/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Њ
ResNet/add/addAddV2,ResNet/leaky_re_lu_2/LeakyRelu:activations:0*ResNet/leaky_re_lu/LeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЭ
+ResNet/batch_normalization_3/ReadVariableOpReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_3/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0√
>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0к
-ResNet/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3ResNet/add/add:z:03ResNet/batch_normalization_3/ReadVariableOp:value:05ResNet/batch_normalization_3/ReadVariableOp_1:value:0DResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ≤
ResNet/leaky_re_lu_3/LeakyRelu	LeakyRelu1ResNet/batch_normalization_3/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Э
%ResNet/conv2d_4/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0с
ResNet/conv2d_4/Conv2DConv2D,ResNet/leaky_re_lu_3/LeakyRelu:activations:0-ResNet/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Т
&ResNet/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
ResNet/conv2d_4/BiasAddBiasAddResNet/conv2d_4/Conv2D:output:0.ResNet/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
+ResNet/batch_normalization_5/ReadVariableOpReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_5/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0у
-ResNet/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_4/BiasAdd:output:03ResNet/batch_normalization_5/ReadVariableOp:value:05ResNet/batch_normalization_5/ReadVariableOp_1:value:0DResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( ±
ResNet/leaky_re_lu_5/LeakyRelu	LeakyRelu1ResNet/batch_normalization_5/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=Э
%ResNet/conv2d_3/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0с
ResNet/conv2d_3/Conv2DConv2D,ResNet/leaky_re_lu_3/LeakyRelu:activations:0-ResNet/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Т
&ResNet/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
ResNet/conv2d_3/BiasAddBiasAddResNet/conv2d_3/Conv2D:output:0.ResNet/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
%ResNet/conv2d_5/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0с
ResNet/conv2d_5/Conv2DConv2D,ResNet/leaky_re_lu_5/LeakyRelu:activations:0-ResNet/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Т
&ResNet/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
ResNet/conv2d_5/BiasAddBiasAddResNet/conv2d_5/Conv2D:output:0.ResNet/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
+ResNet/batch_normalization_4/ReadVariableOpReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_4/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0у
-ResNet/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_3/BiasAdd:output:03ResNet/batch_normalization_4/ReadVariableOp:value:05ResNet/batch_normalization_4/ReadVariableOp_1:value:0DResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( Ь
+ResNet/batch_normalization_6/ReadVariableOpReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_6/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0у
-ResNet/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_5/BiasAdd:output:03ResNet/batch_normalization_6/ReadVariableOp:value:05ResNet/batch_normalization_6/ReadVariableOp_1:value:0DResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( ±
ResNet/leaky_re_lu_6/LeakyRelu	LeakyRelu1ResNet/batch_normalization_6/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=±
ResNet/leaky_re_lu_4/LeakyRelu	LeakyRelu1ResNet/batch_normalization_4/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=Ѕ
ResNet/add_1/addAddV2,ResNet/leaky_re_lu_6/LeakyRelu:activations:0,ResNet/leaky_re_lu_4/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
+ResNet/batch_normalization_7/ReadVariableOpReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_7/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0з
-ResNet/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3ResNet/add_1/add:z:03ResNet/batch_normalization_7/ReadVariableOp:value:05ResNet/batch_normalization_7/ReadVariableOp_1:value:0DResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( ±
ResNet/leaky_re_lu_7/LeakyRelu	LeakyRelu1ResNet/batch_normalization_7/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=Ь
%ResNet/conv2d_6/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0с
ResNet/conv2d_6/Conv2DConv2D,ResNet/leaky_re_lu_7/LeakyRelu:activations:0-ResNet/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
Т
&ResNet/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0њ
ResNet/conv2d_6/BiasAddBiasAddResNet/conv2d_6/Conv2D:output:0.ResNet/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Р
ResNet/conv2d_6/SoftmaxSoftmax ResNet/conv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ћ
 tf.compat.v1.transpose/transpose	Transpose!ResNet/conv2d_6/Softmax:softmax:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Z
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
valueB:м
,random_affine_transform_params/strided_sliceStridedSlice-random_affine_transform_params/Shape:output:0;random_affine_transform_params/strided_slice/stack:output:0=random_affine_transform_params/strided_slice/stack_1:output:0=random_affine_transform_params/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask†
3random_affine_transform_params/random_uniform/shapePack5random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:≈
;random_affine_transform_params/random_uniform/RandomUniformRandomUniform<random_affine_transform_params/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0°
$random_affine_transform_params/RoundRoundDrandom_affine_transform_params/random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:€€€€€€€€€i
$random_affine_transform_params/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @∞
"random_affine_transform_params/mulMul(random_affine_transform_params/Round:y:0-random_affine_transform_params/mul/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€i
$random_affine_transform_params/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ѓ
"random_affine_transform_params/subSub&random_affine_transform_params/mul:z:0-random_affine_transform_params/sub/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ґ
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
 *џIјx
3random_affine_transform_params/random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џI@…
=random_affine_transform_params/random_uniform_1/RandomUniformRandomUniform>random_affine_transform_params/random_uniform_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0„
3random_affine_transform_params/random_uniform_1/subSub<random_affine_transform_params/random_uniform_1/max:output:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*
_output_shapes
: й
3random_affine_transform_params/random_uniform_1/mulMulFrandom_affine_transform_params/random_uniform_1/RandomUniform:output:07random_affine_transform_params/random_uniform_1/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ё
/random_affine_transform_params/random_uniform_1AddV27random_affine_transform_params/random_uniform_1/mul:z:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€М
"random_affine_transform_params/CosCos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€М
"random_affine_transform_params/SinSin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€
"random_affine_transform_params/NegNeg&random_affine_transform_params/Sin:y:0*
T0*#
_output_shapes
:€€€€€€€€€©
$random_affine_transform_params/mul_1Mul&random_affine_transform_params/Neg:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€О
$random_affine_transform_params/Sin_1Sin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€О
$random_affine_transform_params/Cos_1Cos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ђ
$random_affine_transform_params/mul_2Mul(random_affine_transform_params/Cos_1:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€Љ
'random_affine_transform_params/packed/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Њ
'random_affine_transform_params/packed/1Pack(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€–
%random_affine_transform_params/packedPack0random_affine_transform_params/packed/0:output:00random_affine_transform_params/packed/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€В
-random_affine_transform_params/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ”
(random_affine_transform_params/transpose	Transpose.random_affine_transform_params/packed:output:06random_affine_transform_params/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€Њ
)random_affine_transform_params/packed_1/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0*
N*
T0*'
_output_shapes
:€€€€€€€€€ј
)random_affine_transform_params/packed_1/1Pack(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€÷
'random_affine_transform_params/packed_1Pack2random_affine_transform_params/packed_1/0:output:02random_affine_transform_params/packed_1/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Д
/random_affine_transform_params/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ў
*random_affine_transform_params/transpose_1	Transpose0random_affine_transform_params/packed_1:output:08random_affine_transform_params/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€}
$random_affine_transform_params/ConstConst*
_output_shapes

:*
dtype0*!
valueB"у5Cу5C
&random_affine_transform_params/Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   CЋ
%random_affine_transform_params/MatMulBatchMatMulV2,random_affine_transform_params/transpose:y:0/random_affine_transform_params/Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€ј
$random_affine_transform_params/sub_1Sub-random_affine_transform_params/Const:output:0.random_affine_transform_params/MatMul:output:0*
T0*+
_output_shapes
:€€€€€€€€€»
'random_affine_transform_params/MatMul_1BatchMatMulV2.random_affine_transform_params/transpose_1:y:0(random_affine_transform_params/sub_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€Й
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
valueB"         ¶
.random_affine_transform_params/strided_slice_1StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_1/stack:output:0?random_affine_transform_params/strided_slice_1/stack_1:output:0?random_affine_transform_params/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_1Neg7random_affine_transform_params/strided_slice_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€Й
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
valueB"         ¶
.random_affine_transform_params/strided_slice_2StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_2/stack:output:0?random_affine_transform_params/strided_slice_2/stack_1:output:0?random_affine_transform_params/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_2Neg7random_affine_transform_params/strided_slice_2:output:0*
T0*#
_output_shapes
:€€€€€€€€€Г
$random_affine_transform_params/Neg_3Neg(random_affine_transform_params/Neg_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€Ђ
$random_affine_transform_params/mul_3Mul(random_affine_transform_params/Neg_3:y:0&random_affine_transform_params/Cos:y:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/mul_4Mul(random_affine_transform_params/Neg_2:y:0(random_affine_transform_params/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/sub_2Sub(random_affine_transform_params/mul_3:z:0(random_affine_transform_params/mul_4:z:0*
T0*#
_output_shapes
:€€€€€€€€€Г
$random_affine_transform_params/Neg_4Neg(random_affine_transform_params/Neg_2:y:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/mul_5Mul(random_affine_transform_params/Neg_4:y:0(random_affine_transform_params/mul_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/mul_6Mul(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/Sin_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/sub_3Sub(random_affine_transform_params/mul_5:z:0(random_affine_transform_params/mul_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€Е
2random_affine_transform_params/zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€–
,random_affine_transform_params/zeros/ReshapeReshape5random_affine_transform_params/strided_slice:output:0;random_affine_transform_params/zeros/Reshape/shape:output:0*
T0*
_output_shapes
:o
*random_affine_transform_params/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ∆
$random_affine_transform_params/zerosFill5random_affine_transform_params/zeros/Reshape:output:03random_affine_transform_params/zeros/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€З
4random_affine_transform_params/zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€‘
.random_affine_transform_params/zeros_1/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ћ
&random_affine_transform_params/zeros_1Fill7random_affine_transform_params/zeros_1/Reshape:output:05random_affine_transform_params/zeros_1/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ќ
$random_affine_transform_params/stackPack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/sub_2:z:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/sub_3:z:0-random_affine_transform_params/zeros:output:0/random_affine_transform_params/zeros_1:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

axisЗ
4random_affine_transform_params/zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€‘
.random_affine_transform_params/zeros_2/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ћ
&random_affine_transform_params/zeros_2Fill7random_affine_transform_params/zeros_2/Reshape:output:05random_affine_transform_params/zeros_2/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€З
4random_affine_transform_params/zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€‘
.random_affine_transform_params/zeros_3/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ћ
&random_affine_transform_params/zeros_3Fill7random_affine_transform_params/zeros_3/Reshape:output:05random_affine_transform_params/zeros_3/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€—
&random_affine_transform_params/stack_1Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/Neg_2:y:0/random_affine_transform_params/zeros_2:output:0/random_affine_transform_params/zeros_3:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

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
:€€€€€€€€€цц*
dtype0*
interpolation
BILINEARО
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             µ
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ы
(tf.image.adjust_contrast/adjust_contrastAdjustContrastv2Pimage_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:08tf_image_adjust_contrast_adjust_contrast_contrast_factor*1
_output_shapes
:€€€€€€€€€ццђ
1tf.image.adjust_contrast/adjust_contrast/IdentityIdentity1tf.image.adjust_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццЯ
'ResNet/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ф
ResNet/conv2d_1/Conv2D_1Conv2D:tf.image.adjust_contrast/adjust_contrast/Identity:output:0/ResNet/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
Х
(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
ResNet/conv2d_1/BiasAdd_1BiasAdd!ResNet/conv2d_1/Conv2D_1:output:00ResNet/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЯ
-ResNet/batch_normalization_1/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_1/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ѕ
>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0≈
@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
/ResNet/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_1/BiasAdd_1:output:05ResNet/batch_normalization_1/ReadVariableOp_2:value:05ResNet/batch_normalization_1/ReadVariableOp_3:value:0FResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( ¶
 ResNet/leaky_re_lu_1/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_1/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Ы
%ResNet/conv2d/Conv2D_1/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0р
ResNet/conv2d/Conv2D_1Conv2D:tf.image.adjust_contrast/adjust_contrast/Identity:output:0-ResNet/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
С
&ResNet/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0∞
ResNet/conv2d/BiasAdd_1BiasAddResNet/conv2d/Conv2D_1:output:0.ResNet/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА†
'ResNet/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0и
ResNet/conv2d_2/Conv2D_1Conv2D.ResNet/leaky_re_lu_1/LeakyRelu_1:activations:0/ResNet/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
Х
(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
ResNet/conv2d_2/BiasAdd_1BiasAdd!ResNet/conv2d_2/Conv2D_1:output:00ResNet/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЫ
+ResNet/batch_normalization/ReadVariableOp_2ReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0Э
+ResNet/batch_normalization/ReadVariableOp_3ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0љ
<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ѕ
>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ж
-ResNet/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3 ResNet/conv2d/BiasAdd_1:output:03ResNet/batch_normalization/ReadVariableOp_2:value:03ResNet/batch_normalization/ReadVariableOp_3:value:0DResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0FResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( Я
-ResNet/batch_normalization_2/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_2/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ѕ
>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0≈
@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
/ResNet/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_2/BiasAdd_1:output:05ResNet/batch_normalization_2/ReadVariableOp_2:value:05ResNet/batch_normalization_2/ReadVariableOp_3:value:0FResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( ¶
 ResNet/leaky_re_lu_2/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_2/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Ґ
ResNet/leaky_re_lu/LeakyRelu_1	LeakyRelu1ResNet/batch_normalization/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=і
ResNet/add/add_1AddV2.ResNet/leaky_re_lu_2/LeakyRelu_1:activations:0,ResNet/leaky_re_lu/LeakyRelu_1:activations:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЯ
-ResNet/batch_normalization_3/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_3/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ѕ
>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0≈
@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0д
/ResNet/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3ResNet/add/add_1:z:05ResNet/batch_normalization_3/ReadVariableOp_2:value:05ResNet/batch_normalization_3/ReadVariableOp_3:value:0FResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( ¶
 ResNet/leaky_re_lu_3/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_3/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Я
'ResNet/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0з
ResNet/conv2d_4/Conv2D_1Conv2D.ResNet/leaky_re_lu_3/LeakyRelu_1:activations:0/ResNet/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
Ф
(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
ResNet/conv2d_4/BiasAdd_1BiasAdd!ResNet/conv2d_4/Conv2D_1:output:00ResNet/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
-ResNet/batch_normalization_5/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_5/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0ј
>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ƒ
@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
/ResNet/batch_normalization_5/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_4/BiasAdd_1:output:05ResNet/batch_normalization_5/ReadVariableOp_2:value:05ResNet/batch_normalization_5/ReadVariableOp_3:value:0FResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( •
 ResNet/leaky_re_lu_5/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_5/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=Я
'ResNet/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0з
ResNet/conv2d_3/Conv2D_1Conv2D.ResNet/leaky_re_lu_3/LeakyRelu_1:activations:0/ResNet/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
Ф
(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
ResNet/conv2d_3/BiasAdd_1BiasAdd!ResNet/conv2d_3/Conv2D_1:output:00ResNet/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
'ResNet/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0з
ResNet/conv2d_5/Conv2D_1Conv2D.ResNet/leaky_re_lu_5/LeakyRelu_1:activations:0/ResNet/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
Ф
(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
ResNet/conv2d_5/BiasAdd_1BiasAdd!ResNet/conv2d_5/Conv2D_1:output:00ResNet/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
-ResNet/batch_normalization_4/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_4/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0ј
>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ƒ
@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
/ResNet/batch_normalization_4/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_3/BiasAdd_1:output:05ResNet/batch_normalization_4/ReadVariableOp_2:value:05ResNet/batch_normalization_4/ReadVariableOp_3:value:0FResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( Ю
-ResNet/batch_normalization_6/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_6/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0ј
>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ƒ
@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
/ResNet/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_5/BiasAdd_1:output:05ResNet/batch_normalization_6/ReadVariableOp_2:value:05ResNet/batch_normalization_6/ReadVariableOp_3:value:0FResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( •
 ResNet/leaky_re_lu_6/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_6/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=•
 ResNet/leaky_re_lu_4/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_4/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=Ј
ResNet/add_1/add_1AddV2.ResNet/leaky_re_lu_6/LeakyRelu_1:activations:0.ResNet/leaky_re_lu_4/LeakyRelu_1:activations:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
-ResNet/batch_normalization_7/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_7/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0ј
>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ƒ
@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0б
/ResNet/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3ResNet/add_1/add_1:z:05ResNet/batch_normalization_7/ReadVariableOp_2:value:05ResNet/batch_normalization_7/ReadVariableOp_3:value:0FResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( •
 ResNet/leaky_re_lu_7/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_7/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=Ю
'ResNet/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0з
ResNet/conv2d_6/Conv2D_1Conv2D.ResNet/leaky_re_lu_7/LeakyRelu_1:activations:0/ResNet/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц*
paddingSAME*
strides
Ф
(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
ResNet/conv2d_6/BiasAdd_1BiasAdd!ResNet/conv2d_6/Conv2D_1:output:00ResNet/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ццД
ResNet/conv2d_6/Softmax_1Softmax"ResNet/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццЫ
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
 *    і
=image_projective_transform_layer_1/ImageProjectiveTransformV3ImageProjectiveTransformV3#ResNet/conv2d_6/Softmax_1:softmax:0-random_affine_transform_params/stack:output:0Simage_projective_transform_layer_1/ImageProjectiveTransformV3/output_shape:output:0Qimage_projective_transform_layer_1/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
interpolation
BILINEARА
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             с
"tf.compat.v1.transpose_1/transpose	TransposeRimage_projective_transform_layer_1/ImageProjectiveTransformV3:transformed_images:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА€€€€€€€€€Џ
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
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
valueB"            ч
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А@Ц
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ©
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€©
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€≠
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"€€€€ю€€€С
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
value	B :≥
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€a

Identity_1Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: »%
NoOpNoOp;^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp=^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*^ResNet/batch_normalization/ReadVariableOp,^ResNet/batch_normalization/ReadVariableOp_1,^ResNet/batch_normalization/ReadVariableOp_2,^ResNet/batch_normalization/ReadVariableOp_3=^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_1/ReadVariableOp.^ResNet/batch_normalization_1/ReadVariableOp_1.^ResNet/batch_normalization_1/ReadVariableOp_2.^ResNet/batch_normalization_1/ReadVariableOp_3=^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_2/ReadVariableOp.^ResNet/batch_normalization_2/ReadVariableOp_1.^ResNet/batch_normalization_2/ReadVariableOp_2.^ResNet/batch_normalization_2/ReadVariableOp_3=^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_3/ReadVariableOp.^ResNet/batch_normalization_3/ReadVariableOp_1.^ResNet/batch_normalization_3/ReadVariableOp_2.^ResNet/batch_normalization_3/ReadVariableOp_3=^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_4/ReadVariableOp.^ResNet/batch_normalization_4/ReadVariableOp_1.^ResNet/batch_normalization_4/ReadVariableOp_2.^ResNet/batch_normalization_4/ReadVariableOp_3=^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_5/ReadVariableOp.^ResNet/batch_normalization_5/ReadVariableOp_1.^ResNet/batch_normalization_5/ReadVariableOp_2.^ResNet/batch_normalization_5/ReadVariableOp_3=^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_6/ReadVariableOp.^ResNet/batch_normalization_6/ReadVariableOp_1.^ResNet/batch_normalization_6/ReadVariableOp_2.^ResNet/batch_normalization_6/ReadVariableOp_3=^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_7/ReadVariableOp.^ResNet/batch_normalization_7/ReadVariableOp_1.^ResNet/batch_normalization_7/ReadVariableOp_2.^ResNet/batch_normalization_7/ReadVariableOp_3%^ResNet/conv2d/BiasAdd/ReadVariableOp'^ResNet/conv2d/BiasAdd_1/ReadVariableOp$^ResNet/conv2d/Conv2D/ReadVariableOp&^ResNet/conv2d/Conv2D_1/ReadVariableOp'^ResNet/conv2d_1/BiasAdd/ReadVariableOp)^ResNet/conv2d_1/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_1/Conv2D/ReadVariableOp(^ResNet/conv2d_1/Conv2D_1/ReadVariableOp'^ResNet/conv2d_2/BiasAdd/ReadVariableOp)^ResNet/conv2d_2/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_2/Conv2D/ReadVariableOp(^ResNet/conv2d_2/Conv2D_1/ReadVariableOp'^ResNet/conv2d_3/BiasAdd/ReadVariableOp)^ResNet/conv2d_3/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_3/Conv2D/ReadVariableOp(^ResNet/conv2d_3/Conv2D_1/ReadVariableOp'^ResNet/conv2d_4/BiasAdd/ReadVariableOp)^ResNet/conv2d_4/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_4/Conv2D/ReadVariableOp(^ResNet/conv2d_4/Conv2D_1/ReadVariableOp'^ResNet/conv2d_5/BiasAdd/ReadVariableOp)^ResNet/conv2d_5/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_5/Conv2D/ReadVariableOp(^ResNet/conv2d_5/Conv2D_1/ReadVariableOp'^ResNet/conv2d_6/BiasAdd/ReadVariableOp)^ResNet/conv2d_6/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_6/Conv2D/ReadVariableOp(^ResNet/conv2d_6/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2x
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:/

_output_shapes
: 
Е
њ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36524

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
¬
Й
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_36075

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
:€€€€€€€€€АА*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€цц:€€€€€€€€€:Y U
1
_output_shapes
:€€€€€€€€€цц
 
_user_specified_nameinputs:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
transforms
ђ
л

&__inference_ResNet_layer_call_fn_35620

inputs"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:
identityИҐStatefulPartitionedCallƒ
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32531Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
є
Ю
(__inference_conv2d_4_layer_call_fn_36452

inputs"
unknown:А@
	unknown_0:@
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_31956Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
„f
Я
@__inference_model_layer_call_and_return_conditional_losses_33697

inputs'
resnet_33504:А
resnet_33506:	А
resnet_33508:	А
resnet_33510:	А
resnet_33512:	А
resnet_33514:	А'
resnet_33516:А
resnet_33518:	А(
resnet_33520:АА
resnet_33522:	А
resnet_33524:	А
resnet_33526:	А
resnet_33528:	А
resnet_33530:	А
resnet_33532:	А
resnet_33534:	А
resnet_33536:	А
resnet_33538:	А
resnet_33540:	А
resnet_33542:	А
resnet_33544:	А
resnet_33546:	А'
resnet_33548:А@
resnet_33550:@
resnet_33552:@
resnet_33554:@
resnet_33556:@
resnet_33558:@'
resnet_33560:А@
resnet_33562:@&
resnet_33564:@@
resnet_33566:@
resnet_33568:@
resnet_33570:@
resnet_33572:@
resnet_33574:@
resnet_33576:@
resnet_33578:@
resnet_33580:@
resnet_33582:@
resnet_33584:@
resnet_33586:@
resnet_33588:@
resnet_33590:@&
resnet_33592:@
resnet_33594:<
8tf_image_adjust_contrast_adjust_contrast_contrast_factor
identity

identity_1ИҐResNet/StatefulPartitionedCallҐ ResNet/StatefulPartitionedCall_1Ґ6random_affine_transform_params/StatefulPartitionedCallі
ResNet/StatefulPartitionedCallStatefulPartitionedCallinputsresnet_33504resnet_33506resnet_33508resnet_33510resnet_33512resnet_33514resnet_33516resnet_33518resnet_33520resnet_33522resnet_33524resnet_33526resnet_33528resnet_33530resnet_33532resnet_33534resnet_33536resnet_33538resnet_33540resnet_33542resnet_33544resnet_33546resnet_33548resnet_33550resnet_33552resnet_33554resnet_33556resnet_33558resnet_33560resnet_33562resnet_33564resnet_33566resnet_33568resnet_33570resnet_33572resnet_33574resnet_33576resnet_33578resnet_33580resnet_33582resnet_33584resnet_33586resnet_33588resnet_33590resnet_33592resnet_33594*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32531~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             “
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€М
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *b
f]R[
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_33149Є
0image_projective_transform_layer/PartitionedCallPartitionedCallinputs?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_33160О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             µ
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€д
(tf.image.adjust_contrast/adjust_contrastAdjustContrastv29image_projective_transform_layer/PartitionedCall:output:08tf_image_adjust_contrast_adjust_contrast_contrast_factor*1
_output_shapes
:€€€€€€€€€ццђ
1tf.image.adjust_contrast/adjust_contrast/IdentityIdentity1tf.image.adjust_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццы
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall:tf.image.adjust_contrast/adjust_contrast/Identity:output:0resnet_33504resnet_33506resnet_33508resnet_33510resnet_33512resnet_33514resnet_33516resnet_33518resnet_33520resnet_33522resnet_33524resnet_33526resnet_33528resnet_33530resnet_33532resnet_33534resnet_33536resnet_33538resnet_33540resnet_33542resnet_33544resnet_33546resnet_33548resnet_33550resnet_33552resnet_33554resnet_33556resnet_33558resnet_33560resnet_33562resnet_33564resnet_33566resnet_33568resnet_33570resnet_33572resnet_33574resnet_33576resnet_33578resnet_33580resnet_33582resnet_33584resnet_33586resnet_33588resnet_33590resnet_33592resnet_33594^ResNet/StatefulPartitionedCall*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32531я
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_33222А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Џ
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА€€€€€€€€€Џ
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
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
valueB"            ч
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А@Ц
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ©
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€©
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€≠
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"€€€€ю€€€С
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
value	B :≥
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: …
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
GPU2 *0J 8В *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_33267Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: √
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:/

_output_shapes
: 
Ш	
“
3__inference_batch_normalization_layer_call_fn_36290

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31457К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¬
Й
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_33222

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
:€€€€€€€€€АА*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€цц:€€€€€€€€€:Y U
1
_output_shapes
:€€€€€€€€€цц
 
_user_specified_nameinputs:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
transforms
µѕ
М4
__inference__traced_save_37180
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
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop
savev2_const_1

identity_1ИҐMergeV2Checkpointsw
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
: Ч2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:q*
dtype0*ј1
valueґ1B≥1qB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH“
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:q*
dtype0*ч
valueнBкqB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Б2
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *
dtypesu
s2q	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*Ў
_input_shapes∆
√: :А:А:А:А:А:А:АА:А:А:А:А:А:А:А:А:А:А:А:А:А:А:А:А@:@:@:@:@:@:@@:@:А@:@:@:@:@:@:@:@:@:@:@:@:@:@:@:: : : : : : :А:А:А:А:АА:А:А:А:А:А:А:А:А:А:А@:@:@:@:@@:@:А@:@:@:@:@:@:@:@:@::А:А:А:А:АА:А:А:А:А:А:А:А:А:А:А@:@:@:@:@@:@:А@:@:@:@:@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:-	)
'
_output_shapes
:А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:А@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:А@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@:,-(
&
_output_shapes
:@: .

_output_shapes
::/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :-5)
'
_output_shapes
:А:!6

_output_shapes	
:А:!7

_output_shapes	
:А:!8

_output_shapes	
:А:.9*
(
_output_shapes
:АА:!:

_output_shapes	
:А:-;)
'
_output_shapes
:А:!<

_output_shapes	
:А:!=

_output_shapes	
:А:!>

_output_shapes	
:А:!?

_output_shapes	
:А:!@

_output_shapes	
:А:!A

_output_shapes	
:А:!B

_output_shapes	
:А:-C)
'
_output_shapes
:А@: D

_output_shapes
:@: E

_output_shapes
:@: F

_output_shapes
:@:,G(
&
_output_shapes
:@@: H

_output_shapes
:@:-I)
'
_output_shapes
:А@: J

_output_shapes
:@: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@: N

_output_shapes
:@: O

_output_shapes
:@: P

_output_shapes
:@:,Q(
&
_output_shapes
:@: R

_output_shapes
::-S)
'
_output_shapes
:А:!T

_output_shapes	
:А:!U

_output_shapes	
:А:!V

_output_shapes	
:А:.W*
(
_output_shapes
:АА:!X

_output_shapes	
:А:-Y)
'
_output_shapes
:А:!Z

_output_shapes	
:А:![

_output_shapes	
:А:!\

_output_shapes	
:А:!]

_output_shapes	
:А:!^

_output_shapes	
:А:!_

_output_shapes	
:А:!`

_output_shapes	
:А:-a)
'
_output_shapes
:А@: b

_output_shapes
:@: c

_output_shapes
:@: d

_output_shapes
:@:,e(
&
_output_shapes
:@@: f

_output_shapes
:@:-g)
'
_output_shapes
:А@: h

_output_shapes
:@: i

_output_shapes
:@: j

_output_shapes
:@: k

_output_shapes
:@: l

_output_shapes
:@: m

_output_shapes
:@: n

_output_shapes
:@:,o(
&
_output_shapes
:@: p

_output_shapes
::q

_output_shapes
: 
Т	
–
5__inference_batch_normalization_5_layer_call_fn_36488

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31616Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
а
З
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_36061

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
:€€€€€€€€€цц*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:€€€€€€€€€цц"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
transforms
ќ
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_31913

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ъ
ъ

%__inference_model_layer_call_fn_34492

inputs"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45
identityИҐStatefulPartitionedCallг
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
unknown_45*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_33272Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:/

_output_shapes
: 
њ
м

&__inference_ResNet_layer_call_fn_32179
input_2"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:
identityИҐStatefulPartitionedCall’
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32084Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_2
Њ‘
у(
A__inference_ResNet_layer_call_and_return_conditional_losses_35789

inputsB
'conv2d_1_conv2d_readvariableop_resource:А7
(conv2d_1_biasadd_readvariableop_resource:	А<
-batch_normalization_1_readvariableop_resource:	А>
/batch_normalization_1_readvariableop_1_resource:	АM
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	А@
%conv2d_conv2d_readvariableop_resource:А5
&conv2d_biasadd_readvariableop_resource:	АC
'conv2d_2_conv2d_readvariableop_resource:АА7
(conv2d_2_biasadd_readvariableop_resource:	А:
+batch_normalization_readvariableop_resource:	А<
-batch_normalization_readvariableop_1_resource:	АK
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	АM
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	А<
-batch_normalization_2_readvariableop_resource:	А>
/batch_normalization_2_readvariableop_1_resource:	АM
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	А<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АB
'conv2d_4_conv2d_readvariableop_resource:А@6
(conv2d_4_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:А@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_4_readvariableop_resource:@=
/batch_normalization_4_readvariableop_1_resource:@L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@;
-batch_normalization_7_readvariableop_resource:@=
/batch_normalization_7_readvariableop_1_resource:@L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:
identityИҐ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ5batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_4/ReadVariableOpҐ&batch_normalization_4/ReadVariableOp_1Ґ5batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_5/ReadVariableOpҐ&batch_normalization_5/ReadVariableOp_1Ґ5batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_6/ReadVariableOpҐ&batch_normalization_6/ReadVariableOp_1Ґ5batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_7/ReadVariableOpҐ&batch_normalization_7/ReadVariableOp_1Ґconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpП
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0Њ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Е
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АП
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ќ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( §
leaky_re_lu_1/LeakyRelu	LeakyRelu*batch_normalization_1/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Л
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Б
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0•
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АР
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
conv2d_2/Conv2DConv2D%leaky_re_lu_1/LeakyRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Е
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЛ
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0П
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0≠
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0¬
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( П
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ќ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( §
leaky_re_lu_2/LeakyRelu	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=†
leaky_re_lu/LeakyRelu	LeakyRelu(batch_normalization/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=©
add/addAddV2%leaky_re_lu_2/LeakyRelu:activations:0#leaky_re_lu/LeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АП
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ј
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3add/add:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( §
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=П
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0№
conv2d_4/Conv2DConv2D%leaky_re_lu_3/LeakyRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0™
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0…
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( £
leaky_re_lu_5/LeakyRelu	LeakyRelu*batch_normalization_5/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=П
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0№
conv2d_3/Conv2DConv2D%leaky_re_lu_3/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0™
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0№
conv2d_5/Conv2DConv2D%leaky_re_lu_5/LeakyRelu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0™
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0…
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( О
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0…
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( £
leaky_re_lu_6/LeakyRelu	LeakyRelu*batch_normalization_6/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=£
leaky_re_lu_4/LeakyRelu	LeakyRelu*batch_normalization_4/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=ђ
	add_1/addAddV2%leaky_re_lu_6/LeakyRelu:activations:0%leaky_re_lu_4/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0љ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3add_1/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( £
leaky_re_lu_7/LeakyRelu	LeakyRelu*batch_normalization_7/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0№
conv2d_6/Conv2DConv2D%leaky_re_lu_7/LeakyRelu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0™
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€В
conv2d_6/SoftmaxSoftmaxconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityconv2d_6/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Я
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Е
э
C__inference_conv2d_3_layer_call_and_return_conditional_losses_36572

inputs9
conv2d_readvariableop_resource:А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Т	
–
5__inference_batch_normalization_6_layer_call_fn_36598

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31680Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ь	
‘
5__inference_batch_normalization_2_layer_call_fn_36228

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31393К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ћ
Ы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36616

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
і
o
C__inference_add_loss_layer_call_and_return_conditional_losses_33267

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
÷f
Я
@__inference_model_layer_call_and_return_conditional_losses_33272

inputs'
resnet_32972:А
resnet_32974:	А
resnet_32976:	А
resnet_32978:	А
resnet_32980:	А
resnet_32982:	А'
resnet_32984:А
resnet_32986:	А(
resnet_32988:АА
resnet_32990:	А
resnet_32992:	А
resnet_32994:	А
resnet_32996:	А
resnet_32998:	А
resnet_33000:	А
resnet_33002:	А
resnet_33004:	А
resnet_33006:	А
resnet_33008:	А
resnet_33010:	А
resnet_33012:	А
resnet_33014:	А'
resnet_33016:А@
resnet_33018:@
resnet_33020:@
resnet_33022:@
resnet_33024:@
resnet_33026:@'
resnet_33028:А@
resnet_33030:@&
resnet_33032:@@
resnet_33034:@
resnet_33036:@
resnet_33038:@
resnet_33040:@
resnet_33042:@
resnet_33044:@
resnet_33046:@
resnet_33048:@
resnet_33050:@
resnet_33052:@
resnet_33054:@
resnet_33056:@
resnet_33058:@&
resnet_33060:@
resnet_33062:<
8tf_image_adjust_contrast_adjust_contrast_contrast_factor
identity

identity_1ИҐResNet/StatefulPartitionedCallҐ ResNet/StatefulPartitionedCall_1Ґ6random_affine_transform_params/StatefulPartitionedCallƒ
ResNet/StatefulPartitionedCallStatefulPartitionedCallinputsresnet_32972resnet_32974resnet_32976resnet_32978resnet_32980resnet_32982resnet_32984resnet_32986resnet_32988resnet_32990resnet_32992resnet_32994resnet_32996resnet_32998resnet_33000resnet_33002resnet_33004resnet_33006resnet_33008resnet_33010resnet_33012resnet_33014resnet_33016resnet_33018resnet_33020resnet_33022resnet_33024resnet_33026resnet_33028resnet_33030resnet_33032resnet_33034resnet_33036resnet_33038resnet_33040resnet_33042resnet_33044resnet_33046resnet_33048resnet_33050resnet_33052resnet_33054resnet_33056resnet_33058resnet_33060resnet_33062*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32084~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             “
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€М
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *b
f]R[
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_33149Є
0image_projective_transform_layer/PartitionedCallPartitionedCallinputs?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_33160О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             µ
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€д
(tf.image.adjust_contrast/adjust_contrastAdjustContrastv29image_projective_transform_layer/PartitionedCall:output:08tf_image_adjust_contrast_adjust_contrast_contrast_factor*1
_output_shapes
:€€€€€€€€€ццђ
1tf.image.adjust_contrast/adjust_contrast/IdentityIdentity1tf.image.adjust_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццк
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall:tf.image.adjust_contrast/adjust_contrast/Identity:output:0resnet_32972resnet_32974resnet_32976resnet_32978resnet_32980resnet_32982resnet_32984resnet_32986resnet_32988resnet_32990resnet_32992resnet_32994resnet_32996resnet_32998resnet_33000resnet_33002resnet_33004resnet_33006resnet_33008resnet_33010resnet_33012resnet_33014resnet_33016resnet_33018resnet_33020resnet_33022resnet_33024resnet_33026resnet_33028resnet_33030resnet_33032resnet_33034resnet_33036resnet_33038resnet_33040resnet_33042resnet_33044resnet_33046resnet_33048resnet_33050resnet_33052resnet_33054resnet_33056resnet_33058resnet_33060resnet_33062*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32084я
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_33222А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Џ
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА€€€€€€€€€Џ
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
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
valueB"            ч
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А@Ц
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ©
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€©
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€≠
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"€€€€ю€€€С
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
value	B :≥
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: …
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
GPU2 *0J 8В *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_33267Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: √
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:/

_output_shapes
: 
ќ
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_31856

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Е
њ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31680

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И
ю
C__inference_conv2d_1_layer_call_and_return_conditional_losses_36105

inputs9
conv2d_readvariableop_resource:А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ф
ь
C__inference_conv2d_6_layer_call_and_return_conditional_losses_36820

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€p
SoftmaxSoftmaxBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€z
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ъ	
‘
5__inference_batch_normalization_2_layer_call_fn_36241

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31424К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ў
l
@__inference_add_1_layer_call_and_return_conditional_losses_36728
inputs_0
inputs_1
identityl
addAddV2inputs_0inputs_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/1
ґ
Э
(__inference_conv2d_5_layer_call_fn_36543

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_32004Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Е
э
C__inference_conv2d_4_layer_call_and_return_conditional_losses_36462

inputs9
conv2d_readvariableop_resource:А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ї
Q
%__inference_add_1_layer_call_fn_36722
inputs_0
inputs_1
identity„
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_32048z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:k g
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/1
Т
I
-__inference_leaky_re_lu_6_layer_call_fn_36701

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_32033z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
џ
Я
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31521

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ОА
Ј
A__inference_ResNet_layer_call_and_return_conditional_losses_32844
input_2)
conv2d_1_32726:А
conv2d_1_32728:	А*
batch_normalization_1_32731:	А*
batch_normalization_1_32733:	А*
batch_normalization_1_32735:	А*
batch_normalization_1_32737:	А'
conv2d_32741:А
conv2d_32743:	А*
conv2d_2_32746:АА
conv2d_2_32748:	А(
batch_normalization_32751:	А(
batch_normalization_32753:	А(
batch_normalization_32755:	А(
batch_normalization_32757:	А*
batch_normalization_2_32760:	А*
batch_normalization_2_32762:	А*
batch_normalization_2_32764:	А*
batch_normalization_2_32766:	А*
batch_normalization_3_32772:	А*
batch_normalization_3_32774:	А*
batch_normalization_3_32776:	А*
batch_normalization_3_32778:	А)
conv2d_4_32782:А@
conv2d_4_32784:@)
batch_normalization_5_32787:@)
batch_normalization_5_32789:@)
batch_normalization_5_32791:@)
batch_normalization_5_32793:@)
conv2d_3_32797:А@
conv2d_3_32799:@(
conv2d_5_32802:@@
conv2d_5_32804:@)
batch_normalization_4_32807:@)
batch_normalization_4_32809:@)
batch_normalization_4_32811:@)
batch_normalization_4_32813:@)
batch_normalization_6_32816:@)
batch_normalization_6_32818:@)
batch_normalization_6_32820:@)
batch_normalization_6_32822:@)
batch_normalization_7_32828:@)
batch_normalization_7_32830:@)
batch_normalization_7_32832:@)
batch_normalization_7_32834:@(
conv2d_6_32838:@
conv2d_6_32840:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallО
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1_32726conv2d_1_32728*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_31836Ґ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_32731batch_normalization_1_32733batch_normalization_1_32735batch_normalization_1_32737*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31329С
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_31856Ж
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_32741conv2d_32743*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_31868≠
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv2d_2_32746conv2d_2_32748*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_31884Ф
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_32751batch_normalization_32753batch_normalization_32755batch_normalization_32757*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31457Ґ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_32760batch_normalization_2_32762batch_normalization_2_32764batch_normalization_2_32766*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31393С
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_31913Л
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_31920Ф
add/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_31928Х
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_32772batch_normalization_3_32774batch_normalization_3_32776batch_normalization_3_32778*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31521С
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_31944ђ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_4_32782conv2d_4_32784*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_31956°
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_32787batch_normalization_5_32789batch_normalization_5_32791batch_normalization_5_32793*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31585Р
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_31976ђ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_3_32797conv2d_3_32799*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_31988ђ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0conv2d_5_32802conv2d_5_32804*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_32004°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_32807batch_normalization_4_32809batch_normalization_4_32811batch_normalization_4_32813*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31713°
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_32816batch_normalization_6_32818batch_normalization_6_32820batch_normalization_6_32822*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31649Р
leaky_re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_32033Р
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_32040Щ
add_1/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_32048Ц
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_32828batch_normalization_7_32830batch_normalization_7_32832batch_normalization_7_32834*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31777Р
leaky_re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_32064ђ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv2d_6_32838conv2d_6_32840*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_32077Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_2
ѓ
м

&__inference_ResNet_layer_call_fn_32723
input_2"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:
identityИҐStatefulPartitionedCall≈
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32531Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_2
џ
Я
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36415

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ц
I
-__inference_leaky_re_lu_3_layer_call_fn_36438

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_31944{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
оН
п-
A__inference_ResNet_layer_call_and_return_conditional_losses_35958

inputsB
'conv2d_1_conv2d_readvariableop_resource:А7
(conv2d_1_biasadd_readvariableop_resource:	А<
-batch_normalization_1_readvariableop_resource:	А>
/batch_normalization_1_readvariableop_1_resource:	АM
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	А@
%conv2d_conv2d_readvariableop_resource:А5
&conv2d_biasadd_readvariableop_resource:	АC
'conv2d_2_conv2d_readvariableop_resource:АА7
(conv2d_2_biasadd_readvariableop_resource:	А:
+batch_normalization_readvariableop_resource:	А<
-batch_normalization_readvariableop_1_resource:	АK
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	АM
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	А<
-batch_normalization_2_readvariableop_resource:	А>
/batch_normalization_2_readvariableop_1_resource:	АM
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	А<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АB
'conv2d_4_conv2d_readvariableop_resource:А@6
(conv2d_4_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:А@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_4_readvariableop_resource:@=
/batch_normalization_4_readvariableop_1_resource:@L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@;
-batch_normalization_7_readvariableop_resource:@=
/batch_normalization_7_readvariableop_1_resource:@L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:
identityИҐ"batch_normalization/AssignNewValueҐ$batch_normalization/AssignNewValue_1Ґ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ$batch_normalization_1/AssignNewValueҐ&batch_normalization_1/AssignNewValue_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ$batch_normalization_2/AssignNewValueҐ&batch_normalization_2/AssignNewValue_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ$batch_normalization_3/AssignNewValueҐ&batch_normalization_3/AssignNewValue_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ$batch_normalization_4/AssignNewValueҐ&batch_normalization_4/AssignNewValue_1Ґ5batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_4/ReadVariableOpҐ&batch_normalization_4/ReadVariableOp_1Ґ$batch_normalization_5/AssignNewValueҐ&batch_normalization_5/AssignNewValue_1Ґ5batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_5/ReadVariableOpҐ&batch_normalization_5/ReadVariableOp_1Ґ$batch_normalization_6/AssignNewValueҐ&batch_normalization_6/AssignNewValue_1Ґ5batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_6/ReadVariableOpҐ&batch_normalization_6/ReadVariableOp_1Ґ$batch_normalization_7/AssignNewValueҐ&batch_normalization_7/AssignNewValue_1Ґ5batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_7/ReadVariableOpҐ&batch_normalization_7/ReadVariableOp_1Ґconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpП
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0Њ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Е
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АП
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0№
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ю
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(®
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(§
leaky_re_lu_1/LeakyRelu	LeakyRelu*batch_normalization_1/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Л
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0Ї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Б
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0•
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АР
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
conv2d_2/Conv2DConv2D%leaky_re_lu_1/LeakyRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Е
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЛ
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0П
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0≠
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0–
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ц
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(†
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(П
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0№
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ю
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(®
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(§
leaky_re_lu_2/LeakyRelu	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=†
leaky_re_lu/LeakyRelu	LeakyRelu(batch_normalization/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=©
add/addAddV2%leaky_re_lu_2/LeakyRelu:activations:0#leaky_re_lu/LeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АП
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ќ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3add/add:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ю
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(®
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(§
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=П
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0№
conv2d_4/Conv2DConv2D%leaky_re_lu_3/LeakyRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0™
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0„
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ю
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(®
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(£
leaky_re_lu_5/LeakyRelu	LeakyRelu*batch_normalization_5/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=П
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0№
conv2d_3/Conv2DConv2D%leaky_re_lu_3/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0™
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0№
conv2d_5/Conv2DConv2D%leaky_re_lu_5/LeakyRelu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0™
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0„
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ю
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(®
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(О
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0„
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ю
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(®
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(£
leaky_re_lu_6/LeakyRelu	LeakyRelu*batch_normalization_6/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=£
leaky_re_lu_4/LeakyRelu	LeakyRelu*batch_normalization_4/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=ђ
	add_1/addAddV2%leaky_re_lu_6/LeakyRelu:activations:0%leaky_re_lu_4/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@О
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ћ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3add_1/add:z:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ю
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(®
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(£
leaky_re_lu_7/LeakyRelu	LeakyRelu*batch_normalization_7/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0№
conv2d_6/Conv2DConv2D%leaky_re_lu_7/LeakyRelu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0™
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€В
conv2d_6/SoftmaxSoftmaxconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Г
IdentityIdentityconv2d_6/Softmax:softmax:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ы
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
к
ъ

%__inference_model_layer_call_fn_34592

inputs"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45
identityИҐStatefulPartitionedCall”
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
unknown_45*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: *@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_33697Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:/

_output_shapes
: 
ќ
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_36349

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
У
Ѕ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36339

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Э
l
@__inference_image_projective_transform_layer_layer_call_fn_36053

inputs

transforms
identityв
PartitionedCallPartitionedCallinputs
transforms*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_33160j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€цц"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
transforms
љ
O
#__inference_add_layer_call_fn_36365
inputs_0
inputs_1
identity÷
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_31928{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/1
Т
I
-__inference_leaky_re_lu_7_layer_call_fn_36795

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_32064z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Б
ь
C__inference_conv2d_5_layer_call_and_return_conditional_losses_32004

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
У
Ѕ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31488

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
 
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_32040

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ф	
–
5__inference_batch_normalization_6_layer_call_fn_36585

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31649Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ґ
Э
&__inference_conv2d_layer_call_fn_36205

inputs"
unknown:А
	unknown_0:	А
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_31868К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ь	
‘
5__inference_batch_normalization_1_layer_call_fn_36118

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31329К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Х
√
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31360

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
№f
†
@__inference_model_layer_call_and_return_conditional_losses_34287
input_1'
resnet_34094:А
resnet_34096:	А
resnet_34098:	А
resnet_34100:	А
resnet_34102:	А
resnet_34104:	А'
resnet_34106:А
resnet_34108:	А(
resnet_34110:АА
resnet_34112:	А
resnet_34114:	А
resnet_34116:	А
resnet_34118:	А
resnet_34120:	А
resnet_34122:	А
resnet_34124:	А
resnet_34126:	А
resnet_34128:	А
resnet_34130:	А
resnet_34132:	А
resnet_34134:	А
resnet_34136:	А'
resnet_34138:А@
resnet_34140:@
resnet_34142:@
resnet_34144:@
resnet_34146:@
resnet_34148:@'
resnet_34150:А@
resnet_34152:@&
resnet_34154:@@
resnet_34156:@
resnet_34158:@
resnet_34160:@
resnet_34162:@
resnet_34164:@
resnet_34166:@
resnet_34168:@
resnet_34170:@
resnet_34172:@
resnet_34174:@
resnet_34176:@
resnet_34178:@
resnet_34180:@&
resnet_34182:@
resnet_34184:<
8tf_image_adjust_contrast_adjust_contrast_contrast_factor
identity

identity_1ИҐResNet/StatefulPartitionedCallҐ ResNet/StatefulPartitionedCall_1Ґ6random_affine_transform_params/StatefulPartitionedCallµ
ResNet/StatefulPartitionedCallStatefulPartitionedCallinput_1resnet_34094resnet_34096resnet_34098resnet_34100resnet_34102resnet_34104resnet_34106resnet_34108resnet_34110resnet_34112resnet_34114resnet_34116resnet_34118resnet_34120resnet_34122resnet_34124resnet_34126resnet_34128resnet_34130resnet_34132resnet_34134resnet_34136resnet_34138resnet_34140resnet_34142resnet_34144resnet_34146resnet_34148resnet_34150resnet_34152resnet_34154resnet_34156resnet_34158resnet_34160resnet_34162resnet_34164resnet_34166resnet_34168resnet_34170resnet_34172resnet_34174resnet_34176resnet_34178resnet_34180resnet_34182resnet_34184*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32531~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             “
 tf.compat.v1.transpose/transpose	Transpose'ResNet/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
6random_affine_transform_params/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *b
f]R[
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_33149є
0image_projective_transform_layer/PartitionedCallPartitionedCallinput_1?random_affine_transform_params/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *d
f_R]
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_33160О
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             µ
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€д
(tf.image.adjust_contrast/adjust_contrastAdjustContrastv29image_projective_transform_layer/PartitionedCall:output:08tf_image_adjust_contrast_adjust_contrast_contrast_factor*1
_output_shapes
:€€€€€€€€€ццђ
1tf.image.adjust_contrast/adjust_contrast/IdentityIdentity1tf.image.adjust_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццы
 ResNet/StatefulPartitionedCall_1StatefulPartitionedCall:tf.image.adjust_contrast/adjust_contrast/Identity:output:0resnet_34094resnet_34096resnet_34098resnet_34100resnet_34102resnet_34104resnet_34106resnet_34108resnet_34110resnet_34112resnet_34114resnet_34116resnet_34118resnet_34120resnet_34122resnet_34124resnet_34126resnet_34128resnet_34130resnet_34132resnet_34134resnet_34136resnet_34138resnet_34140resnet_34142resnet_34144resnet_34146resnet_34148resnet_34150resnet_34152resnet_34154resnet_34156resnet_34158resnet_34160resnet_34162resnet_34164resnet_34166resnet_34168resnet_34170resnet_34172resnet_34174resnet_34176resnet_34178resnet_34180resnet_34182resnet_34184^ResNet/StatefulPartitionedCall*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€цц*@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32531я
2image_projective_transform_layer_1/PartitionedCallPartitionedCall)ResNet/StatefulPartitionedCall_1:output:0?random_affine_transform_params/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *f
faR_
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_33222А
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Џ
"tf.compat.v1.transpose_1/transpose	Transpose;image_projective_transform_layer_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА€€€€€€€€€Џ
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
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
valueB"            ч
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А@Ц
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ©
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€©
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€≠
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"€€€€ю€€€С
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
value	B :≥
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*
_output_shapes
: И
tf.math.reduce_mean/MeanMean!tf.math.reduce_sum_2/Sum:output:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: …
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
GPU2 *0J 8В *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_33267Р
IdentityIdentity'ResNet/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€a

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: √
NoOpNoOp^ResNet/StatefulPartitionedCall!^ResNet/StatefulPartitionedCall_17^random_affine_transform_params/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
ResNet/StatefulPartitionedCallResNet/StatefulPartitionedCall2D
 ResNet/StatefulPartitionedCall_1 ResNet/StatefulPartitionedCall_12p
6random_affine_transform_params/StatefulPartitionedCall6random_affine_transform_params/StatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:/

_output_shapes
: 
і
o
C__inference_add_loss_layer_call_and_return_conditional_losses_36086

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
Ћ
Ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36772

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И
ю
C__inference_conv2d_1_layer_call_and_return_conditional_losses_31836

inputs9
conv2d_readvariableop_resource:А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
н
ы

%__inference_model_layer_call_fn_33895
input_1"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45
identityИҐStatefulPartitionedCall‘
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
unknown_44
unknown_45*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: *@
_read_only_resource_inputs"
 	
 !"%&)*-.*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_33697Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:/

_output_shapes
: 
 
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_36716

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ґ
Э
(__inference_conv2d_6_layer_call_fn_36809

inputs!
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_32077Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ф	
–
5__inference_batch_normalization_5_layer_call_fn_36475

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31585Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
 
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_32033

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ћ
Ы
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31713

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ХЪ
ЊG
 __inference__wrapped_model_31307
input_1O
4model_resnet_conv2d_1_conv2d_readvariableop_resource:АD
5model_resnet_conv2d_1_biasadd_readvariableop_resource:	АI
:model_resnet_batch_normalization_1_readvariableop_resource:	АK
<model_resnet_batch_normalization_1_readvariableop_1_resource:	АZ
Kmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	А\
Mmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	АM
2model_resnet_conv2d_conv2d_readvariableop_resource:АB
3model_resnet_conv2d_biasadd_readvariableop_resource:	АP
4model_resnet_conv2d_2_conv2d_readvariableop_resource:ААD
5model_resnet_conv2d_2_biasadd_readvariableop_resource:	АG
8model_resnet_batch_normalization_readvariableop_resource:	АI
:model_resnet_batch_normalization_readvariableop_1_resource:	АX
Imodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:	АZ
Kmodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	АI
:model_resnet_batch_normalization_2_readvariableop_resource:	АK
<model_resnet_batch_normalization_2_readvariableop_1_resource:	АZ
Kmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	А\
Mmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	АI
:model_resnet_batch_normalization_3_readvariableop_resource:	АK
<model_resnet_batch_normalization_3_readvariableop_1_resource:	АZ
Kmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	А\
Mmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АO
4model_resnet_conv2d_4_conv2d_readvariableop_resource:А@C
5model_resnet_conv2d_4_biasadd_readvariableop_resource:@H
:model_resnet_batch_normalization_5_readvariableop_resource:@J
<model_resnet_batch_normalization_5_readvariableop_1_resource:@Y
Kmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@[
Mmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@O
4model_resnet_conv2d_3_conv2d_readvariableop_resource:А@C
5model_resnet_conv2d_3_biasadd_readvariableop_resource:@N
4model_resnet_conv2d_5_conv2d_readvariableop_resource:@@C
5model_resnet_conv2d_5_biasadd_readvariableop_resource:@H
:model_resnet_batch_normalization_4_readvariableop_resource:@J
<model_resnet_batch_normalization_4_readvariableop_1_resource:@Y
Kmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@[
Mmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@H
:model_resnet_batch_normalization_6_readvariableop_resource:@J
<model_resnet_batch_normalization_6_readvariableop_1_resource:@Y
Kmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@[
Mmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@H
:model_resnet_batch_normalization_7_readvariableop_resource:@J
<model_resnet_batch_normalization_7_readvariableop_1_resource:@Y
Kmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@[
Mmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@N
4model_resnet_conv2d_6_conv2d_readvariableop_resource:@C
5model_resnet_conv2d_6_biasadd_readvariableop_resource:B
>model_tf_image_adjust_contrast_adjust_contrast_contrast_factor
identityИҐ@model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpҐBmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ҐBmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpҐDmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Ґ/model/ResNet/batch_normalization/ReadVariableOpҐ1model/ResNet/batch_normalization/ReadVariableOp_1Ґ1model/ResNet/batch_normalization/ReadVariableOp_2Ґ1model/ResNet/batch_normalization/ReadVariableOp_3ҐBmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐDmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ҐDmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpҐFmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Ґ1model/ResNet/batch_normalization_1/ReadVariableOpҐ3model/ResNet/batch_normalization_1/ReadVariableOp_1Ґ3model/ResNet/batch_normalization_1/ReadVariableOp_2Ґ3model/ResNet/batch_normalization_1/ReadVariableOp_3ҐBmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐDmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ҐDmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpҐFmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1Ґ1model/ResNet/batch_normalization_2/ReadVariableOpҐ3model/ResNet/batch_normalization_2/ReadVariableOp_1Ґ3model/ResNet/batch_normalization_2/ReadVariableOp_2Ґ3model/ResNet/batch_normalization_2/ReadVariableOp_3ҐBmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐDmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ҐDmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpҐFmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1Ґ1model/ResNet/batch_normalization_3/ReadVariableOpҐ3model/ResNet/batch_normalization_3/ReadVariableOp_1Ґ3model/ResNet/batch_normalization_3/ReadVariableOp_2Ґ3model/ResNet/batch_normalization_3/ReadVariableOp_3ҐBmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐDmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ҐDmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpҐFmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1Ґ1model/ResNet/batch_normalization_4/ReadVariableOpҐ3model/ResNet/batch_normalization_4/ReadVariableOp_1Ґ3model/ResNet/batch_normalization_4/ReadVariableOp_2Ґ3model/ResNet/batch_normalization_4/ReadVariableOp_3ҐBmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐDmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ҐDmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpҐFmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1Ґ1model/ResNet/batch_normalization_5/ReadVariableOpҐ3model/ResNet/batch_normalization_5/ReadVariableOp_1Ґ3model/ResNet/batch_normalization_5/ReadVariableOp_2Ґ3model/ResNet/batch_normalization_5/ReadVariableOp_3ҐBmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐDmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ҐDmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpҐFmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1Ґ1model/ResNet/batch_normalization_6/ReadVariableOpҐ3model/ResNet/batch_normalization_6/ReadVariableOp_1Ґ3model/ResNet/batch_normalization_6/ReadVariableOp_2Ґ3model/ResNet/batch_normalization_6/ReadVariableOp_3ҐBmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐDmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ҐDmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpҐFmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1Ґ1model/ResNet/batch_normalization_7/ReadVariableOpҐ3model/ResNet/batch_normalization_7/ReadVariableOp_1Ґ3model/ResNet/batch_normalization_7/ReadVariableOp_2Ґ3model/ResNet/batch_normalization_7/ReadVariableOp_3Ґ*model/ResNet/conv2d/BiasAdd/ReadVariableOpҐ,model/ResNet/conv2d/BiasAdd_1/ReadVariableOpҐ)model/ResNet/conv2d/Conv2D/ReadVariableOpҐ+model/ResNet/conv2d/Conv2D_1/ReadVariableOpҐ,model/ResNet/conv2d_1/BiasAdd/ReadVariableOpҐ.model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOpҐ+model/ResNet/conv2d_1/Conv2D/ReadVariableOpҐ-model/ResNet/conv2d_1/Conv2D_1/ReadVariableOpҐ,model/ResNet/conv2d_2/BiasAdd/ReadVariableOpҐ.model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOpҐ+model/ResNet/conv2d_2/Conv2D/ReadVariableOpҐ-model/ResNet/conv2d_2/Conv2D_1/ReadVariableOpҐ,model/ResNet/conv2d_3/BiasAdd/ReadVariableOpҐ.model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOpҐ+model/ResNet/conv2d_3/Conv2D/ReadVariableOpҐ-model/ResNet/conv2d_3/Conv2D_1/ReadVariableOpҐ,model/ResNet/conv2d_4/BiasAdd/ReadVariableOpҐ.model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOpҐ+model/ResNet/conv2d_4/Conv2D/ReadVariableOpҐ-model/ResNet/conv2d_4/Conv2D_1/ReadVariableOpҐ,model/ResNet/conv2d_5/BiasAdd/ReadVariableOpҐ.model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOpҐ+model/ResNet/conv2d_5/Conv2D/ReadVariableOpҐ-model/ResNet/conv2d_5/Conv2D_1/ReadVariableOpҐ,model/ResNet/conv2d_6/BiasAdd/ReadVariableOpҐ.model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOpҐ+model/ResNet/conv2d_6/Conv2D/ReadVariableOpҐ-model/ResNet/conv2d_6/Conv2D_1/ReadVariableOp©
+model/ResNet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ў
model/ResNet/conv2d_1/Conv2DConv2Dinput_13model/ResNet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Я
,model/ResNet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0“
model/ResNet/conv2d_1/BiasAddBiasAdd%model/ResNet/conv2d_1/Conv2D:output:04model/ResNet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А©
1model/ResNet/batch_normalization_1/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≠
3model/ResNet/batch_normalization_1/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ћ
Bmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0ѕ
Dmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ь
3model/ResNet/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_1/BiasAdd:output:09model/ResNet/batch_normalization_1/ReadVariableOp:value:0;model/ResNet/batch_normalization_1/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( Њ
$model/ResNet/leaky_re_lu_1/LeakyRelu	LeakyRelu7model/ResNet/batch_normalization_1/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=•
)model/ResNet/conv2d/Conv2D/ReadVariableOpReadVariableOp2model_resnet_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0’
model/ResNet/conv2d/Conv2DConv2Dinput_11model/ResNet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Ы
*model/ResNet/conv2d/BiasAdd/ReadVariableOpReadVariableOp3model_resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ћ
model/ResNet/conv2d/BiasAddBiasAdd#model/ResNet/conv2d/Conv2D:output:02model/ResNet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А™
+model/ResNet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Д
model/ResNet/conv2d_2/Conv2DConv2D2model/ResNet/leaky_re_lu_1/LeakyRelu:activations:03model/ResNet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
Я
,model/ResNet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0“
model/ResNet/conv2d_2/BiasAddBiasAdd%model/ResNet/conv2d_2/Conv2D:output:04model/ResNet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А•
/model/ResNet/batch_normalization/ReadVariableOpReadVariableOp8model_resnet_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0©
1model/ResNet/batch_normalization/ReadVariableOp_1ReadVariableOp:model_resnet_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0«
@model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpImodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ћ
Bmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKmodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Р
1model/ResNet/batch_normalization/FusedBatchNormV3FusedBatchNormV3$model/ResNet/conv2d/BiasAdd:output:07model/ResNet/batch_normalization/ReadVariableOp:value:09model/ResNet/batch_normalization/ReadVariableOp_1:value:0Hmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jmodel/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ©
1model/ResNet/batch_normalization_2/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0≠
3model/ResNet/batch_normalization_2/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ћ
Bmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0ѕ
Dmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ь
3model/ResNet/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_2/BiasAdd:output:09model/ResNet/batch_normalization_2/ReadVariableOp:value:0;model/ResNet/batch_normalization_2/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( Њ
$model/ResNet/leaky_re_lu_2/LeakyRelu	LeakyRelu7model/ResNet/batch_normalization_2/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Ї
"model/ResNet/leaky_re_lu/LeakyRelu	LeakyRelu5model/ResNet/batch_normalization/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=–
model/ResNet/add/addAddV22model/ResNet/leaky_re_lu_2/LeakyRelu:activations:00model/ResNet/leaky_re_lu/LeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А©
1model/ResNet/batch_normalization_3/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0≠
3model/ResNet/batch_normalization_3/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ћ
Bmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0ѕ
Dmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0О
3model/ResNet/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3model/ResNet/add/add:z:09model/ResNet/batch_normalization_3/ReadVariableOp:value:0;model/ResNet/batch_normalization_3/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( Њ
$model/ResNet/leaky_re_lu_3/LeakyRelu	LeakyRelu7model/ResNet/batch_normalization_3/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=©
+model/ResNet/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Г
model/ResNet/conv2d_4/Conv2DConv2D2model/ResNet/leaky_re_lu_3/LeakyRelu:activations:03model/ResNet/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Ю
,model/ResNet/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0—
model/ResNet/conv2d_4/BiasAddBiasAdd%model/ResNet/conv2d_4/Conv2D:output:04model/ResNet/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@®
1model/ResNet/batch_normalization_5/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_5/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0 
Bmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
Dmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
3model/ResNet/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_4/BiasAdd:output:09model/ResNet/batch_normalization_5/ReadVariableOp:value:0;model/ResNet/batch_normalization_5/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( љ
$model/ResNet/leaky_re_lu_5/LeakyRelu	LeakyRelu7model/ResNet/batch_normalization_5/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=©
+model/ResNet/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Г
model/ResNet/conv2d_3/Conv2DConv2D2model/ResNet/leaky_re_lu_3/LeakyRelu:activations:03model/ResNet/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Ю
,model/ResNet/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0—
model/ResNet/conv2d_3/BiasAddBiasAdd%model/ResNet/conv2d_3/Conv2D:output:04model/ResNet/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@®
+model/ResNet/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Г
model/ResNet/conv2d_5/Conv2DConv2D2model/ResNet/leaky_re_lu_5/LeakyRelu:activations:03model/ResNet/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Ю
,model/ResNet/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0—
model/ResNet/conv2d_5/BiasAddBiasAdd%model/ResNet/conv2d_5/Conv2D:output:04model/ResNet/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@®
1model/ResNet/batch_normalization_4/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_4/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0 
Bmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
Dmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
3model/ResNet/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_3/BiasAdd:output:09model/ResNet/batch_normalization_4/ReadVariableOp:value:0;model/ResNet/batch_normalization_4/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( ®
1model/ResNet/batch_normalization_6/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_6/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0 
Bmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
Dmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
3model/ResNet/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3&model/ResNet/conv2d_5/BiasAdd:output:09model/ResNet/batch_normalization_6/ReadVariableOp:value:0;model/ResNet/batch_normalization_6/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( љ
$model/ResNet/leaky_re_lu_6/LeakyRelu	LeakyRelu7model/ResNet/batch_normalization_6/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=љ
$model/ResNet/leaky_re_lu_4/LeakyRelu	LeakyRelu7model/ResNet/batch_normalization_4/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=”
model/ResNet/add_1/addAddV22model/ResNet/leaky_re_lu_6/LeakyRelu:activations:02model/ResNet/leaky_re_lu_4/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@®
1model/ResNet/batch_normalization_7/ReadVariableOpReadVariableOp:model_resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_7/ReadVariableOp_1ReadVariableOp<model_resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0 
Bmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ќ
Dmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Л
3model/ResNet/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3model/ResNet/add_1/add:z:09model/ResNet/batch_normalization_7/ReadVariableOp:value:0;model/ResNet/batch_normalization_7/ReadVariableOp_1:value:0Jmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( љ
$model/ResNet/leaky_re_lu_7/LeakyRelu	LeakyRelu7model/ResNet/batch_normalization_7/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=®
+model/ResNet/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4model_resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Г
model/ResNet/conv2d_6/Conv2DConv2D2model/ResNet/leaky_re_lu_7/LeakyRelu:activations:03model/ResNet/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
Ю
,model/ResNet/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5model_resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0—
model/ResNet/conv2d_6/BiasAddBiasAdd%model/ResNet/conv2d_6/Conv2D:output:04model/ResNet/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ь
model/ResNet/conv2d_6/SoftmaxSoftmax&model/ResNet/conv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Д
+model/tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ё
&model/tf.compat.v1.transpose/transpose	Transpose'model/ResNet/conv2d_6/Softmax:softmax:04model/tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€a
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
shrink_axis_maskђ
9model/random_affine_transform_params/random_uniform/shapePack;model/random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:—
Amodel/random_affine_transform_params/random_uniform/RandomUniformRandomUniformBmodel/random_affine_transform_params/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0≠
*model/random_affine_transform_params/RoundRoundJmodel/random_affine_transform_params/random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:€€€€€€€€€o
*model/random_affine_transform_params/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¬
(model/random_affine_transform_params/mulMul.model/random_affine_transform_params/Round:y:03model/random_affine_transform_params/mul/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€o
*model/random_affine_transform_params/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ј
(model/random_affine_transform_params/subSub,model/random_affine_transform_params/mul:z:03model/random_affine_transform_params/sub/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ѓ
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
 *џIј~
9model/random_affine_transform_params/random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џI@’
Cmodel/random_affine_transform_params/random_uniform_1/RandomUniformRandomUniformDmodel/random_affine_transform_params/random_uniform_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0й
9model/random_affine_transform_params/random_uniform_1/subSubBmodel/random_affine_transform_params/random_uniform_1/max:output:0Bmodel/random_affine_transform_params/random_uniform_1/min:output:0*
T0*
_output_shapes
: ы
9model/random_affine_transform_params/random_uniform_1/mulMulLmodel/random_affine_transform_params/random_uniform_1/RandomUniform:output:0=model/random_affine_transform_params/random_uniform_1/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€п
5model/random_affine_transform_params/random_uniform_1AddV2=model/random_affine_transform_params/random_uniform_1/mul:z:0Bmodel/random_affine_transform_params/random_uniform_1/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ш
(model/random_affine_transform_params/CosCos9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ш
(model/random_affine_transform_params/SinSin9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€Л
(model/random_affine_transform_params/NegNeg,model/random_affine_transform_params/Sin:y:0*
T0*#
_output_shapes
:€€€€€€€€€ї
*model/random_affine_transform_params/mul_1Mul,model/random_affine_transform_params/Neg:y:0,model/random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ъ
*model/random_affine_transform_params/Sin_1Sin9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ъ
*model/random_affine_transform_params/Cos_1Cos9model/random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€љ
*model/random_affine_transform_params/mul_2Mul.model/random_affine_transform_params/Cos_1:y:0,model/random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€ќ
-model/random_affine_transform_params/packed/0Pack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/mul_1:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€–
-model/random_affine_transform_params/packed/1Pack.model/random_affine_transform_params/Sin_1:y:0.model/random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€в
+model/random_affine_transform_params/packedPack6model/random_affine_transform_params/packed/0:output:06model/random_affine_transform_params/packed/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€И
3model/random_affine_transform_params/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          е
.model/random_affine_transform_params/transpose	Transpose4model/random_affine_transform_params/packed:output:0<model/random_affine_transform_params/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€–
/model/random_affine_transform_params/packed_1/0Pack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/Sin_1:y:0*
N*
T0*'
_output_shapes
:€€€€€€€€€“
/model/random_affine_transform_params/packed_1/1Pack.model/random_affine_transform_params/mul_1:z:0.model/random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€и
-model/random_affine_transform_params/packed_1Pack8model/random_affine_transform_params/packed_1/0:output:08model/random_affine_transform_params/packed_1/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€К
5model/random_affine_transform_params/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          л
0model/random_affine_transform_params/transpose_1	Transpose6model/random_affine_transform_params/packed_1:output:0>model/random_affine_transform_params/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€Г
*model/random_affine_transform_params/ConstConst*
_output_shapes

:*
dtype0*!
valueB"у5Cу5CЕ
,model/random_affine_transform_params/Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   CЁ
+model/random_affine_transform_params/MatMulBatchMatMulV22model/random_affine_transform_params/transpose:y:05model/random_affine_transform_params/Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€“
*model/random_affine_transform_params/sub_1Sub3model/random_affine_transform_params/Const:output:04model/random_affine_transform_params/MatMul:output:0*
T0*+
_output_shapes
:€€€€€€€€€Џ
-model/random_affine_transform_params/MatMul_1BatchMatMulV24model/random_affine_transform_params/transpose_1:y:0.model/random_affine_transform_params/sub_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€П
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
valueB"         ƒ
4model/random_affine_transform_params/strided_slice_1StridedSlice6model/random_affine_transform_params/MatMul_1:output:0Cmodel/random_affine_transform_params/strided_slice_1/stack:output:0Emodel/random_affine_transform_params/strided_slice_1/stack_1:output:0Emodel/random_affine_transform_params/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskЮ
*model/random_affine_transform_params/Neg_1Neg=model/random_affine_transform_params/strided_slice_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€П
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
valueB"         ƒ
4model/random_affine_transform_params/strided_slice_2StridedSlice6model/random_affine_transform_params/MatMul_1:output:0Cmodel/random_affine_transform_params/strided_slice_2/stack:output:0Emodel/random_affine_transform_params/strided_slice_2/stack_1:output:0Emodel/random_affine_transform_params/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskЮ
*model/random_affine_transform_params/Neg_2Neg=model/random_affine_transform_params/strided_slice_2:output:0*
T0*#
_output_shapes
:€€€€€€€€€П
*model/random_affine_transform_params/Neg_3Neg.model/random_affine_transform_params/Neg_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€љ
*model/random_affine_transform_params/mul_3Mul.model/random_affine_transform_params/Neg_3:y:0,model/random_affine_transform_params/Cos:y:0*
T0*#
_output_shapes
:€€€€€€€€€њ
*model/random_affine_transform_params/mul_4Mul.model/random_affine_transform_params/Neg_2:y:0.model/random_affine_transform_params/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€њ
*model/random_affine_transform_params/sub_2Sub.model/random_affine_transform_params/mul_3:z:0.model/random_affine_transform_params/mul_4:z:0*
T0*#
_output_shapes
:€€€€€€€€€П
*model/random_affine_transform_params/Neg_4Neg.model/random_affine_transform_params/Neg_2:y:0*
T0*#
_output_shapes
:€€€€€€€€€њ
*model/random_affine_transform_params/mul_5Mul.model/random_affine_transform_params/Neg_4:y:0.model/random_affine_transform_params/mul_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€њ
*model/random_affine_transform_params/mul_6Mul.model/random_affine_transform_params/Neg_1:y:0.model/random_affine_transform_params/Sin_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€њ
*model/random_affine_transform_params/sub_3Sub.model/random_affine_transform_params/mul_5:z:0.model/random_affine_transform_params/mul_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€Л
8model/random_affine_transform_params/zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€в
2model/random_affine_transform_params/zeros/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Amodel/random_affine_transform_params/zeros/Reshape/shape:output:0*
T0*
_output_shapes
:u
0model/random_affine_transform_params/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ў
*model/random_affine_transform_params/zerosFill;model/random_affine_transform_params/zeros/Reshape:output:09model/random_affine_transform_params/zeros/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€Н
:model/random_affine_transform_params/zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€ж
4model/random_affine_transform_params/zeros_1/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Cmodel/random_affine_transform_params/zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:w
2model/random_affine_transform_params/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
,model/random_affine_transform_params/zeros_1Fill=model/random_affine_transform_params/zeros_1/Reshape:output:0;model/random_affine_transform_params/zeros_1/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€Г
*model/random_affine_transform_params/stackPack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/mul_1:z:0.model/random_affine_transform_params/sub_2:z:0.model/random_affine_transform_params/Sin_1:y:0.model/random_affine_transform_params/mul_2:z:0.model/random_affine_transform_params/sub_3:z:03model/random_affine_transform_params/zeros:output:05model/random_affine_transform_params/zeros_1:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

axisН
:model/random_affine_transform_params/zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€ж
4model/random_affine_transform_params/zeros_2/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Cmodel/random_affine_transform_params/zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:w
2model/random_affine_transform_params/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
,model/random_affine_transform_params/zeros_2Fill=model/random_affine_transform_params/zeros_2/Reshape:output:0;model/random_affine_transform_params/zeros_2/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€Н
:model/random_affine_transform_params/zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€ж
4model/random_affine_transform_params/zeros_3/ReshapeReshape;model/random_affine_transform_params/strided_slice:output:0Cmodel/random_affine_transform_params/zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:w
2model/random_affine_transform_params/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
,model/random_affine_transform_params/zeros_3Fill=model/random_affine_transform_params/zeros_3/Reshape:output:0;model/random_affine_transform_params/zeros_3/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€З
,model/random_affine_transform_params/stack_1Pack,model/random_affine_transform_params/Cos:y:0.model/random_affine_transform_params/Sin_1:y:0.model/random_affine_transform_params/Neg_1:y:0.model/random_affine_transform_params/mul_1:z:0.model/random_affine_transform_params/mul_2:z:0.model/random_affine_transform_params/Neg_2:y:05model/random_affine_transform_params/zeros_2:output:05model/random_affine_transform_params/zeros_3:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

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
 *    ђ
Amodel/image_projective_transform_layer/ImageProjectiveTransformV3ImageProjectiveTransformV3input_15model/random_affine_transform_params/stack_1:output:0Wmodel/image_projective_transform_layer/ImageProjectiveTransformV3/output_shape:output:0Umodel/image_projective_transform_layer/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:€€€€€€€€€цц*
dtype0*
interpolation
BILINEARФ
#model/tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             «
model/tf.compat.v1.pad/PadPad*model/tf.compat.v1.transpose/transpose:y:0,model/tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
.model/tf.image.adjust_contrast/adjust_contrastAdjustContrastv2Vmodel/image_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:0>model_tf_image_adjust_contrast_adjust_contrast_contrast_factor*1
_output_shapes
:€€€€€€€€€ццЄ
7model/tf.image.adjust_contrast/adjust_contrast/IdentityIdentity7model/tf.image.adjust_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццЂ
-model/ResNet/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0Ж
model/ResNet/conv2d_1/Conv2D_1Conv2D@model/tf.image.adjust_contrast/adjust_contrast/Identity:output:05model/ResNet/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
°
.model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0»
model/ResNet/conv2d_1/BiasAdd_1BiasAdd'model/ResNet/conv2d_1/Conv2D_1:output:06model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЂ
3model/ResNet/batch_normalization_1/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≠
3model/ResNet/batch_normalization_1/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
Dmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0—
Fmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ц
5model/ResNet/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_1/BiasAdd_1:output:0;model/ResNet/batch_normalization_1/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_1/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( ≤
&model/ResNet/leaky_re_lu_1/LeakyRelu_1	LeakyRelu9model/ResNet/batch_normalization_1/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=І
+model/ResNet/conv2d/Conv2D_1/ReadVariableOpReadVariableOp2model_resnet_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0В
model/ResNet/conv2d/Conv2D_1Conv2D@model/tf.image.adjust_contrast/adjust_contrast/Identity:output:03model/ResNet/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
Э
,model/ResNet/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp3model_resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0¬
model/ResNet/conv2d/BiasAdd_1BiasAdd%model/ResNet/conv2d/Conv2D_1:output:04model/ResNet/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАђ
-model/ResNet/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ъ
model/ResNet/conv2d_2/Conv2D_1Conv2D4model/ResNet/leaky_re_lu_1/LeakyRelu_1:activations:05model/ResNet/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
°
.model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0»
model/ResNet/conv2d_2/BiasAdd_1BiasAdd'model/ResNet/conv2d_2/Conv2D_1:output:06model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАІ
1model/ResNet/batch_normalization/ReadVariableOp_2ReadVariableOp8model_resnet_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0©
1model/ResNet/batch_normalization/ReadVariableOp_3ReadVariableOp:model_resnet_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0…
Bmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpImodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ќ
Dmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpKmodel_resnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0К
3model/ResNet/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3&model/ResNet/conv2d/BiasAdd_1:output:09model/ResNet/batch_normalization/ReadVariableOp_2:value:09model/ResNet/batch_normalization/ReadVariableOp_3:value:0Jmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0Lmodel/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( Ђ
3model/ResNet/batch_normalization_2/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0≠
3model/ResNet/batch_normalization_2/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
Dmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0—
Fmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ц
5model/ResNet/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_2/BiasAdd_1:output:0;model/ResNet/batch_normalization_2/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_2/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( ≤
&model/ResNet/leaky_re_lu_2/LeakyRelu_1	LeakyRelu9model/ResNet/batch_normalization_2/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Ѓ
$model/ResNet/leaky_re_lu/LeakyRelu_1	LeakyRelu7model/ResNet/batch_normalization/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=∆
model/ResNet/add/add_1AddV24model/ResNet/leaky_re_lu_2/LeakyRelu_1:activations:02model/ResNet/leaky_re_lu/LeakyRelu_1:activations:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЂ
3model/ResNet/batch_normalization_3/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0≠
3model/ResNet/batch_normalization_3/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
Dmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0—
Fmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0И
5model/ResNet/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3model/ResNet/add/add_1:z:0;model/ResNet/batch_normalization_3/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_3/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
is_training( ≤
&model/ResNet/leaky_re_lu_3/LeakyRelu_1	LeakyRelu9model/ResNet/batch_normalization_3/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Ђ
-model/ResNet/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0щ
model/ResNet/conv2d_4/Conv2D_1Conv2D4model/ResNet/leaky_re_lu_3/LeakyRelu_1:activations:05model/ResNet/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
†
.model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0«
model/ResNet/conv2d_4/BiasAdd_1BiasAdd'model/ResNet/conv2d_4/Conv2D_1:output:06model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@™
3model/ResNet/batch_normalization_5/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_5/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
Dmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0–
Fmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0С
5model/ResNet/batch_normalization_5/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_4/BiasAdd_1:output:0;model/ResNet/batch_normalization_5/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_5/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( ±
&model/ResNet/leaky_re_lu_5/LeakyRelu_1	LeakyRelu9model/ResNet/batch_normalization_5/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=Ђ
-model/ResNet/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0щ
model/ResNet/conv2d_3/Conv2D_1Conv2D4model/ResNet/leaky_re_lu_3/LeakyRelu_1:activations:05model/ResNet/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
†
.model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0«
model/ResNet/conv2d_3/BiasAdd_1BiasAdd'model/ResNet/conv2d_3/Conv2D_1:output:06model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@™
-model/ResNet/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0щ
model/ResNet/conv2d_5/Conv2D_1Conv2D4model/ResNet/leaky_re_lu_5/LeakyRelu_1:activations:05model/ResNet/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
†
.model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0«
model/ResNet/conv2d_5/BiasAdd_1BiasAdd'model/ResNet/conv2d_5/Conv2D_1:output:06model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@™
3model/ResNet/batch_normalization_4/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_4/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
Dmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0–
Fmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0С
5model/ResNet/batch_normalization_4/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_3/BiasAdd_1:output:0;model/ResNet/batch_normalization_4/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_4/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( ™
3model/ResNet/batch_normalization_6/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_6/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
Dmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0–
Fmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0С
5model/ResNet/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3(model/ResNet/conv2d_5/BiasAdd_1:output:0;model/ResNet/batch_normalization_6/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_6/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( ±
&model/ResNet/leaky_re_lu_6/LeakyRelu_1	LeakyRelu9model/ResNet/batch_normalization_6/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=±
&model/ResNet/leaky_re_lu_4/LeakyRelu_1	LeakyRelu9model/ResNet/batch_normalization_4/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=…
model/ResNet/add_1/add_1AddV24model/ResNet/leaky_re_lu_6/LeakyRelu_1:activations:04model/ResNet/leaky_re_lu_4/LeakyRelu_1:activations:0*
T0*1
_output_shapes
:€€€€€€€€€цц@™
3model/ResNet/batch_normalization_7/ReadVariableOp_2ReadVariableOp:model_resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0ђ
3model/ResNet/batch_normalization_7/ReadVariableOp_3ReadVariableOp<model_resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
Dmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpKmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0–
Fmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpMmodel_resnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Е
5model/ResNet/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3model/ResNet/add_1/add_1:z:0;model/ResNet/batch_normalization_7/ReadVariableOp_2:value:0;model/ResNet/batch_normalization_7/ReadVariableOp_3:value:0Lmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0Nmodel/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
is_training( ±
&model/ResNet/leaky_re_lu_7/LeakyRelu_1	LeakyRelu9model/ResNet/batch_normalization_7/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=™
-model/ResNet/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp4model_resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0щ
model/ResNet/conv2d_6/Conv2D_1Conv2D4model/ResNet/leaky_re_lu_7/LeakyRelu_1:activations:05model/ResNet/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц*
paddingSAME*
strides
†
.model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp5model_resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0«
model/ResNet/conv2d_6/BiasAdd_1BiasAdd'model/ResNet/conv2d_6/Conv2D_1:output:06model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ццР
model/ResNet/conv2d_6/Softmax_1Softmax(model/ResNet/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€цц°
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
 *    “
Cmodel/image_projective_transform_layer_1/ImageProjectiveTransformV3ImageProjectiveTransformV3)model/ResNet/conv2d_6/Softmax_1:softmax:03model/random_affine_transform_params/stack:output:0Ymodel/image_projective_transform_layer_1/ImageProjectiveTransformV3/output_shape:output:0Wmodel/image_projective_transform_layer_1/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
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
:АА€€€€€€€€€м
#model/tf.compat.v1.nn.conv2d/Conv2DConv2D#model/tf.compat.v1.pad/Pad:output:0,model/tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
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

:*

begin_mask	*
end_mask	*
shrink_axis_maskН
"model/tf.compat.v1.squeeze/SqueezeSqueeze5model/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:d
model/tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А@®
model/tf.math.truediv/truedivRealDiv+model/tf.compat.v1.squeeze/Squeeze:output:0(model/tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:f
!model/tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCҐ
model/tf.math.truediv_1/truedivRealDiv!model/tf.math.truediv/truediv:z:0*model/tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:f
!model/tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АC§
model/tf.math.truediv_2/truedivRealDiv#model/tf.math.truediv_1/truediv:z:0*model/tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:~
-model/tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ї
(model/tf.compat.v1.transpose_2/transpose	Transpose#model/tf.math.truediv_2/truediv:z:06model/tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:•
 model/tf.__operators__.add/AddV2AddV2#model/tf.math.truediv_2/truediv:z:0,model/tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:f
!model/tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @•
model/tf.math.truediv_3/truedivRealDiv$model/tf.__operators__.add/AddV2:z:0*model/tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:c
model/tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Ґ
"model/tf.__operators__.add_2/AddV2AddV2#model/tf.math.truediv_3/truediv:z:0'model/tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:y
.model/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ї
model/tf.math.reduce_sum/SumSum#model/tf.math.truediv_3/truediv:z:07model/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims({
0model/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€њ
model/tf.math.reduce_sum_1/SumSum#model/tf.math.truediv_3/truediv:z:09model/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Ъ
model/tf.math.multiply/MulMul%model/tf.math.reduce_sum/Sum:output:0'model/tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:c
model/tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Э
"model/tf.__operators__.add_1/AddV2AddV2model/tf.math.multiply/Mul:z:0'model/tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:£
model/tf.math.truediv_4/truedivRealDiv&model/tf.__operators__.add_1/AddV2:z:0&model/tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:j
model/tf.math.log/LogLog#model/tf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:М
model/tf.math.multiply_1/MulMul#model/tf.math.truediv_3/truediv:z:0model/tf.math.log/Log:y:0*
T0*
_output_shapes

:Б
0model/tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"€€€€ю€€€£
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
value	B :Ћ
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€р)
NoOpNoOpA^model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpC^model/ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1C^model/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpE^model/ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_10^model/ResNet/batch_normalization/ReadVariableOp2^model/ResNet/batch_normalization/ReadVariableOp_12^model/ResNet/batch_normalization/ReadVariableOp_22^model/ResNet/batch_normalization/ReadVariableOp_3C^model/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_1/ReadVariableOp4^model/ResNet/batch_normalization_1/ReadVariableOp_14^model/ResNet/batch_normalization_1/ReadVariableOp_24^model/ResNet/batch_normalization_1/ReadVariableOp_3C^model/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_2/ReadVariableOp4^model/ResNet/batch_normalization_2/ReadVariableOp_14^model/ResNet/batch_normalization_2/ReadVariableOp_24^model/ResNet/batch_normalization_2/ReadVariableOp_3C^model/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_3/ReadVariableOp4^model/ResNet/batch_normalization_3/ReadVariableOp_14^model/ResNet/batch_normalization_3/ReadVariableOp_24^model/ResNet/batch_normalization_3/ReadVariableOp_3C^model/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_4/ReadVariableOp4^model/ResNet/batch_normalization_4/ReadVariableOp_14^model/ResNet/batch_normalization_4/ReadVariableOp_24^model/ResNet/batch_normalization_4/ReadVariableOp_3C^model/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_5/ReadVariableOp4^model/ResNet/batch_normalization_5/ReadVariableOp_14^model/ResNet/batch_normalization_5/ReadVariableOp_24^model/ResNet/batch_normalization_5/ReadVariableOp_3C^model/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_6/ReadVariableOp4^model/ResNet/batch_normalization_6/ReadVariableOp_14^model/ResNet/batch_normalization_6/ReadVariableOp_24^model/ResNet/batch_normalization_6/ReadVariableOp_3C^model/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^model/ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1E^model/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpG^model/ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12^model/ResNet/batch_normalization_7/ReadVariableOp4^model/ResNet/batch_normalization_7/ReadVariableOp_14^model/ResNet/batch_normalization_7/ReadVariableOp_24^model/ResNet/batch_normalization_7/ReadVariableOp_3+^model/ResNet/conv2d/BiasAdd/ReadVariableOp-^model/ResNet/conv2d/BiasAdd_1/ReadVariableOp*^model/ResNet/conv2d/Conv2D/ReadVariableOp,^model/ResNet/conv2d/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_1/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_1/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_1/Conv2D/ReadVariableOp.^model/ResNet/conv2d_1/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_2/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_2/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_2/Conv2D/ReadVariableOp.^model/ResNet/conv2d_2/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_3/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_3/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_3/Conv2D/ReadVariableOp.^model/ResNet/conv2d_3/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_4/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_4/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_4/Conv2D/ReadVariableOp.^model/ResNet/conv2d_4/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_5/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_5/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_5/Conv2D/ReadVariableOp.^model/ResNet/conv2d_5/Conv2D_1/ReadVariableOp-^model/ResNet/conv2d_6/BiasAdd/ReadVariableOp/^model/ResNet/conv2d_6/BiasAdd_1/ReadVariableOp,^model/ResNet/conv2d_6/Conv2D/ReadVariableOp.^model/ResNet/conv2d_6/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Д
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:/

_output_shapes
: 
а
З
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_33160

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
:€€€€€€€€€цц*
dtype0*
interpolation
BILINEARБ
IdentityIdentity/ImageProjectiveTransformV3:transformed_images:0*
T0*1
_output_shapes
:€€€€€€€€€цц"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:€€€€€€€€€:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
transforms
ў
Э
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36321

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
’
h
>__inference_add_layer_call_and_return_conditional_losses_31928

inputs
inputs_1
identityk
addAddV2inputsinputs_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аj
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Т
I
-__inference_leaky_re_lu_5_layer_call_fn_36529

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_31976z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
“ф
УM
@__inference_model_layer_call_and_return_conditional_losses_35426

inputsI
.resnet_conv2d_1_conv2d_readvariableop_resource:А>
/resnet_conv2d_1_biasadd_readvariableop_resource:	АC
4resnet_batch_normalization_1_readvariableop_resource:	АE
6resnet_batch_normalization_1_readvariableop_1_resource:	АT
Eresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АV
Gresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	АG
,resnet_conv2d_conv2d_readvariableop_resource:А<
-resnet_conv2d_biasadd_readvariableop_resource:	АJ
.resnet_conv2d_2_conv2d_readvariableop_resource:АА>
/resnet_conv2d_2_biasadd_readvariableop_resource:	АA
2resnet_batch_normalization_readvariableop_resource:	АC
4resnet_batch_normalization_readvariableop_1_resource:	АR
Cresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:	АT
Eresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	АC
4resnet_batch_normalization_2_readvariableop_resource:	АE
6resnet_batch_normalization_2_readvariableop_1_resource:	АT
Eresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АV
Gresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	АC
4resnet_batch_normalization_3_readvariableop_resource:	АE
6resnet_batch_normalization_3_readvariableop_1_resource:	АT
Eresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АV
Gresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АI
.resnet_conv2d_4_conv2d_readvariableop_resource:А@=
/resnet_conv2d_4_biasadd_readvariableop_resource:@B
4resnet_batch_normalization_5_readvariableop_resource:@D
6resnet_batch_normalization_5_readvariableop_1_resource:@S
Eresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@I
.resnet_conv2d_3_conv2d_readvariableop_resource:А@=
/resnet_conv2d_3_biasadd_readvariableop_resource:@H
.resnet_conv2d_5_conv2d_readvariableop_resource:@@=
/resnet_conv2d_5_biasadd_readvariableop_resource:@B
4resnet_batch_normalization_4_readvariableop_resource:@D
6resnet_batch_normalization_4_readvariableop_1_resource:@S
Eresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@B
4resnet_batch_normalization_6_readvariableop_resource:@D
6resnet_batch_normalization_6_readvariableop_1_resource:@S
Eresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@B
4resnet_batch_normalization_7_readvariableop_resource:@D
6resnet_batch_normalization_7_readvariableop_1_resource:@S
Eresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:@U
Gresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:@H
.resnet_conv2d_6_conv2d_readvariableop_resource:@=
/resnet_conv2d_6_biasadd_readvariableop_resource:<
8tf_image_adjust_contrast_adjust_contrast_contrast_factor
identity

identity_1ИҐ)ResNet/batch_normalization/AssignNewValueҐ+ResNet/batch_normalization/AssignNewValue_1Ґ+ResNet/batch_normalization/AssignNewValue_2Ґ+ResNet/batch_normalization/AssignNewValue_3Ґ:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpҐ>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1Ґ)ResNet/batch_normalization/ReadVariableOpҐ+ResNet/batch_normalization/ReadVariableOp_1Ґ+ResNet/batch_normalization/ReadVariableOp_2Ґ+ResNet/batch_normalization/ReadVariableOp_3Ґ+ResNet/batch_normalization_1/AssignNewValueҐ-ResNet/batch_normalization_1/AssignNewValue_1Ґ-ResNet/batch_normalization_1/AssignNewValue_2Ґ-ResNet/batch_normalization_1/AssignNewValue_3Ґ<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_1/ReadVariableOpҐ-ResNet/batch_normalization_1/ReadVariableOp_1Ґ-ResNet/batch_normalization_1/ReadVariableOp_2Ґ-ResNet/batch_normalization_1/ReadVariableOp_3Ґ+ResNet/batch_normalization_2/AssignNewValueҐ-ResNet/batch_normalization_2/AssignNewValue_1Ґ-ResNet/batch_normalization_2/AssignNewValue_2Ґ-ResNet/batch_normalization_2/AssignNewValue_3Ґ<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_2/ReadVariableOpҐ-ResNet/batch_normalization_2/ReadVariableOp_1Ґ-ResNet/batch_normalization_2/ReadVariableOp_2Ґ-ResNet/batch_normalization_2/ReadVariableOp_3Ґ+ResNet/batch_normalization_3/AssignNewValueҐ-ResNet/batch_normalization_3/AssignNewValue_1Ґ-ResNet/batch_normalization_3/AssignNewValue_2Ґ-ResNet/batch_normalization_3/AssignNewValue_3Ґ<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_3/ReadVariableOpҐ-ResNet/batch_normalization_3/ReadVariableOp_1Ґ-ResNet/batch_normalization_3/ReadVariableOp_2Ґ-ResNet/batch_normalization_3/ReadVariableOp_3Ґ+ResNet/batch_normalization_4/AssignNewValueҐ-ResNet/batch_normalization_4/AssignNewValue_1Ґ-ResNet/batch_normalization_4/AssignNewValue_2Ґ-ResNet/batch_normalization_4/AssignNewValue_3Ґ<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_4/ReadVariableOpҐ-ResNet/batch_normalization_4/ReadVariableOp_1Ґ-ResNet/batch_normalization_4/ReadVariableOp_2Ґ-ResNet/batch_normalization_4/ReadVariableOp_3Ґ+ResNet/batch_normalization_5/AssignNewValueҐ-ResNet/batch_normalization_5/AssignNewValue_1Ґ-ResNet/batch_normalization_5/AssignNewValue_2Ґ-ResNet/batch_normalization_5/AssignNewValue_3Ґ<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_5/ReadVariableOpҐ-ResNet/batch_normalization_5/ReadVariableOp_1Ґ-ResNet/batch_normalization_5/ReadVariableOp_2Ґ-ResNet/batch_normalization_5/ReadVariableOp_3Ґ+ResNet/batch_normalization_6/AssignNewValueҐ-ResNet/batch_normalization_6/AssignNewValue_1Ґ-ResNet/batch_normalization_6/AssignNewValue_2Ґ-ResNet/batch_normalization_6/AssignNewValue_3Ґ<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_6/ReadVariableOpҐ-ResNet/batch_normalization_6/ReadVariableOp_1Ґ-ResNet/batch_normalization_6/ReadVariableOp_2Ґ-ResNet/batch_normalization_6/ReadVariableOp_3Ґ+ResNet/batch_normalization_7/AssignNewValueҐ-ResNet/batch_normalization_7/AssignNewValue_1Ґ-ResNet/batch_normalization_7/AssignNewValue_2Ґ-ResNet/batch_normalization_7/AssignNewValue_3Ґ<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐ>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpҐ@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1Ґ+ResNet/batch_normalization_7/ReadVariableOpҐ-ResNet/batch_normalization_7/ReadVariableOp_1Ґ-ResNet/batch_normalization_7/ReadVariableOp_2Ґ-ResNet/batch_normalization_7/ReadVariableOp_3Ґ$ResNet/conv2d/BiasAdd/ReadVariableOpҐ&ResNet/conv2d/BiasAdd_1/ReadVariableOpҐ#ResNet/conv2d/Conv2D/ReadVariableOpҐ%ResNet/conv2d/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_1/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_1/Conv2D/ReadVariableOpҐ'ResNet/conv2d_1/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_2/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_2/Conv2D/ReadVariableOpҐ'ResNet/conv2d_2/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_3/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_3/Conv2D/ReadVariableOpҐ'ResNet/conv2d_3/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_4/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_4/Conv2D/ReadVariableOpҐ'ResNet/conv2d_4/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_5/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_5/Conv2D/ReadVariableOpҐ'ResNet/conv2d_5/Conv2D_1/ReadVariableOpҐ&ResNet/conv2d_6/BiasAdd/ReadVariableOpҐ(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpҐ%ResNet/conv2d_6/Conv2D/ReadVariableOpҐ'ResNet/conv2d_6/Conv2D_1/ReadVariableOpЭ
%ResNet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ћ
ResNet/conv2d_1/Conv2DConv2Dinputs-ResNet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
У
&ResNet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ј
ResNet/conv2d_1/BiasAddBiasAddResNet/conv2d_1/Conv2D:output:0.ResNet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЭ
+ResNet/batch_normalization_1/ReadVariableOpReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_1/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0√
>ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ж
-ResNet/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_1/BiasAdd:output:03ResNet/batch_normalization_1/ReadVariableOp:value:05ResNet/batch_normalization_1/ReadVariableOp_1:value:0DResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ї
+ResNet/batch_normalization_1/AssignNewValueAssignVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_1/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ƒ
-ResNet/batch_normalization_1/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_1/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(≤
ResNet/leaky_re_lu_1/LeakyRelu	LeakyRelu1ResNet/batch_normalization_1/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Щ
#ResNet/conv2d/Conv2D/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0»
ResNet/conv2d/Conv2DConv2Dinputs+ResNet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
П
$ResNet/conv2d/BiasAdd/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ї
ResNet/conv2d/BiasAddBiasAddResNet/conv2d/Conv2D:output:0,ResNet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЮ
%ResNet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0т
ResNet/conv2d_2/Conv2DConv2D,ResNet/leaky_re_lu_1/LeakyRelu:activations:0-ResNet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
У
&ResNet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ј
ResNet/conv2d_2/BiasAddBiasAddResNet/conv2d_2/Conv2D:output:0.ResNet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЩ
)ResNet/batch_normalization/ReadVariableOpReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0Э
+ResNet/batch_normalization/ReadVariableOp_1ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ї
:ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ъ
+ResNet/batch_normalization/FusedBatchNormV3FusedBatchNormV3ResNet/conv2d/BiasAdd:output:01ResNet/batch_normalization/ReadVariableOp:value:03ResNet/batch_normalization/ReadVariableOp_1:value:0BResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0DResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<≤
)ResNet/batch_normalization/AssignNewValueAssignVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource8ResNet/batch_normalization/FusedBatchNormV3:batch_mean:0;^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Љ
+ResNet/batch_normalization/AssignNewValue_1AssignVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource<ResNet/batch_normalization/FusedBatchNormV3:batch_variance:0=^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Э
+ResNet/batch_normalization_2/ReadVariableOpReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_2/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0√
>ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ж
-ResNet/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_2/BiasAdd:output:03ResNet/batch_normalization_2/ReadVariableOp:value:05ResNet/batch_normalization_2/ReadVariableOp_1:value:0DResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ї
+ResNet/batch_normalization_2/AssignNewValueAssignVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_2/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ƒ
-ResNet/batch_normalization_2/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_2/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(≤
ResNet/leaky_re_lu_2/LeakyRelu	LeakyRelu1ResNet/batch_normalization_2/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Ѓ
ResNet/leaky_re_lu/LeakyRelu	LeakyRelu/ResNet/batch_normalization/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Њ
ResNet/add/addAddV2,ResNet/leaky_re_lu_2/LeakyRelu:activations:0*ResNet/leaky_re_lu/LeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЭ
+ResNet/batch_normalization_3/ReadVariableOpReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_3/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0њ
<ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0√
>ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ш
-ResNet/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3ResNet/add/add:z:03ResNet/batch_normalization_3/ReadVariableOp:value:05ResNet/batch_normalization_3/ReadVariableOp_1:value:0DResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ї
+ResNet/batch_normalization_3/AssignNewValueAssignVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_3/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ƒ
-ResNet/batch_normalization_3/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_3/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(≤
ResNet/leaky_re_lu_3/LeakyRelu	LeakyRelu1ResNet/batch_normalization_3/FusedBatchNormV3:y:0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=Э
%ResNet/conv2d_4/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0с
ResNet/conv2d_4/Conv2DConv2D,ResNet/leaky_re_lu_3/LeakyRelu:activations:0-ResNet/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Т
&ResNet/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
ResNet/conv2d_4/BiasAddBiasAddResNet/conv2d_4/Conv2D:output:0.ResNet/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
+ResNet/batch_normalization_5/ReadVariableOpReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_5/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Б
-ResNet/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_4/BiasAdd:output:03ResNet/batch_normalization_5/ReadVariableOp:value:05ResNet/batch_normalization_5/ReadVariableOp_1:value:0DResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ї
+ResNet/batch_normalization_5/AssignNewValueAssignVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_5/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ƒ
-ResNet/batch_normalization_5/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_5/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(±
ResNet/leaky_re_lu_5/LeakyRelu	LeakyRelu1ResNet/batch_normalization_5/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=Э
%ResNet/conv2d_3/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0с
ResNet/conv2d_3/Conv2DConv2D,ResNet/leaky_re_lu_3/LeakyRelu:activations:0-ResNet/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Т
&ResNet/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
ResNet/conv2d_3/BiasAddBiasAddResNet/conv2d_3/Conv2D:output:0.ResNet/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
%ResNet/conv2d_5/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0с
ResNet/conv2d_5/Conv2DConv2D,ResNet/leaky_re_lu_5/LeakyRelu:activations:0-ResNet/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
Т
&ResNet/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
ResNet/conv2d_5/BiasAddBiasAddResNet/conv2d_5/Conv2D:output:0.ResNet/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
+ResNet/batch_normalization_4/ReadVariableOpReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_4/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Б
-ResNet/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_3/BiasAdd:output:03ResNet/batch_normalization_4/ReadVariableOp:value:05ResNet/batch_normalization_4/ReadVariableOp_1:value:0DResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ї
+ResNet/batch_normalization_4/AssignNewValueAssignVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_4/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ƒ
-ResNet/batch_normalization_4/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_4/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ь
+ResNet/batch_normalization_6/ReadVariableOpReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_6/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Б
-ResNet/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 ResNet/conv2d_5/BiasAdd:output:03ResNet/batch_normalization_6/ReadVariableOp:value:05ResNet/batch_normalization_6/ReadVariableOp_1:value:0DResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ї
+ResNet/batch_normalization_6/AssignNewValueAssignVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_6/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ƒ
-ResNet/batch_normalization_6/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_6/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(±
ResNet/leaky_re_lu_6/LeakyRelu	LeakyRelu1ResNet/batch_normalization_6/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=±
ResNet/leaky_re_lu_4/LeakyRelu	LeakyRelu1ResNet/batch_normalization_4/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=Ѕ
ResNet/add_1/addAddV2,ResNet/leaky_re_lu_6/LeakyRelu:activations:0,ResNet/leaky_re_lu_4/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
+ResNet/batch_normalization_7/ReadVariableOpReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_7/ReadVariableOp_1ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
<ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¬
>ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0х
-ResNet/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3ResNet/add_1/add:z:03ResNet/batch_normalization_7/ReadVariableOp:value:05ResNet/batch_normalization_7/ReadVariableOp_1:value:0DResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0FResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ї
+ResNet/batch_normalization_7/AssignNewValueAssignVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization_7/FusedBatchNormV3:batch_mean:0=^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ƒ
-ResNet/batch_normalization_7/AssignNewValue_1AssignVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization_7/FusedBatchNormV3:batch_variance:0?^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(±
ResNet/leaky_re_lu_7/LeakyRelu	LeakyRelu1ResNet/batch_normalization_7/FusedBatchNormV3:y:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=Ь
%ResNet/conv2d_6/Conv2D/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0с
ResNet/conv2d_6/Conv2DConv2D,ResNet/leaky_re_lu_7/LeakyRelu:activations:0-ResNet/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingSAME*
strides
Т
&ResNet/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0њ
ResNet/conv2d_6/BiasAddBiasAddResNet/conv2d_6/Conv2D:output:0.ResNet/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Р
ResNet/conv2d_6/SoftmaxSoftmax ResNet/conv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ћ
 tf.compat.v1.transpose/transpose	Transpose!ResNet/conv2d_6/Softmax:softmax:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Z
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
valueB:м
,random_affine_transform_params/strided_sliceStridedSlice-random_affine_transform_params/Shape:output:0;random_affine_transform_params/strided_slice/stack:output:0=random_affine_transform_params/strided_slice/stack_1:output:0=random_affine_transform_params/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask†
3random_affine_transform_params/random_uniform/shapePack5random_affine_transform_params/strided_slice:output:0*
N*
T0*
_output_shapes
:≈
;random_affine_transform_params/random_uniform/RandomUniformRandomUniform<random_affine_transform_params/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0°
$random_affine_transform_params/RoundRoundDrandom_affine_transform_params/random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:€€€€€€€€€i
$random_affine_transform_params/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @∞
"random_affine_transform_params/mulMul(random_affine_transform_params/Round:y:0-random_affine_transform_params/mul/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€i
$random_affine_transform_params/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ѓ
"random_affine_transform_params/subSub&random_affine_transform_params/mul:z:0-random_affine_transform_params/sub/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ґ
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
 *џIјx
3random_affine_transform_params/random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џI@…
=random_affine_transform_params/random_uniform_1/RandomUniformRandomUniform>random_affine_transform_params/random_uniform_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0„
3random_affine_transform_params/random_uniform_1/subSub<random_affine_transform_params/random_uniform_1/max:output:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*
_output_shapes
: й
3random_affine_transform_params/random_uniform_1/mulMulFrandom_affine_transform_params/random_uniform_1/RandomUniform:output:07random_affine_transform_params/random_uniform_1/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ё
/random_affine_transform_params/random_uniform_1AddV27random_affine_transform_params/random_uniform_1/mul:z:0<random_affine_transform_params/random_uniform_1/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€М
"random_affine_transform_params/CosCos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€М
"random_affine_transform_params/SinSin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€
"random_affine_transform_params/NegNeg&random_affine_transform_params/Sin:y:0*
T0*#
_output_shapes
:€€€€€€€€€©
$random_affine_transform_params/mul_1Mul&random_affine_transform_params/Neg:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€О
$random_affine_transform_params/Sin_1Sin3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€О
$random_affine_transform_params/Cos_1Cos3random_affine_transform_params/random_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€Ђ
$random_affine_transform_params/mul_2Mul(random_affine_transform_params/Cos_1:y:0&random_affine_transform_params/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€Љ
'random_affine_transform_params/packed/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Њ
'random_affine_transform_params/packed/1Pack(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€–
%random_affine_transform_params/packedPack0random_affine_transform_params/packed/0:output:00random_affine_transform_params/packed/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€В
-random_affine_transform_params/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ”
(random_affine_transform_params/transpose	Transpose.random_affine_transform_params/packed:output:06random_affine_transform_params/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€Њ
)random_affine_transform_params/packed_1/0Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0*
N*
T0*'
_output_shapes
:€€€€€€€€€ј
)random_affine_transform_params/packed_1/1Pack(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€÷
'random_affine_transform_params/packed_1Pack2random_affine_transform_params/packed_1/0:output:02random_affine_transform_params/packed_1/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Д
/random_affine_transform_params/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ў
*random_affine_transform_params/transpose_1	Transpose0random_affine_transform_params/packed_1:output:08random_affine_transform_params/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€}
$random_affine_transform_params/ConstConst*
_output_shapes

:*
dtype0*!
valueB"у5Cу5C
&random_affine_transform_params/Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   CЋ
%random_affine_transform_params/MatMulBatchMatMulV2,random_affine_transform_params/transpose:y:0/random_affine_transform_params/Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€ј
$random_affine_transform_params/sub_1Sub-random_affine_transform_params/Const:output:0.random_affine_transform_params/MatMul:output:0*
T0*+
_output_shapes
:€€€€€€€€€»
'random_affine_transform_params/MatMul_1BatchMatMulV2.random_affine_transform_params/transpose_1:y:0(random_affine_transform_params/sub_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€Й
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
valueB"         ¶
.random_affine_transform_params/strided_slice_1StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_1/stack:output:0?random_affine_transform_params/strided_slice_1/stack_1:output:0?random_affine_transform_params/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_1Neg7random_affine_transform_params/strided_slice_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€Й
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
valueB"         ¶
.random_affine_transform_params/strided_slice_2StridedSlice0random_affine_transform_params/MatMul_1:output:0=random_affine_transform_params/strided_slice_2/stack:output:0?random_affine_transform_params/strided_slice_2/stack_1:output:0?random_affine_transform_params/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskТ
$random_affine_transform_params/Neg_2Neg7random_affine_transform_params/strided_slice_2:output:0*
T0*#
_output_shapes
:€€€€€€€€€Г
$random_affine_transform_params/Neg_3Neg(random_affine_transform_params/Neg_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€Ђ
$random_affine_transform_params/mul_3Mul(random_affine_transform_params/Neg_3:y:0&random_affine_transform_params/Cos:y:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/mul_4Mul(random_affine_transform_params/Neg_2:y:0(random_affine_transform_params/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/sub_2Sub(random_affine_transform_params/mul_3:z:0(random_affine_transform_params/mul_4:z:0*
T0*#
_output_shapes
:€€€€€€€€€Г
$random_affine_transform_params/Neg_4Neg(random_affine_transform_params/Neg_2:y:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/mul_5Mul(random_affine_transform_params/Neg_4:y:0(random_affine_transform_params/mul_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/mul_6Mul(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/Sin_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€≠
$random_affine_transform_params/sub_3Sub(random_affine_transform_params/mul_5:z:0(random_affine_transform_params/mul_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€Е
2random_affine_transform_params/zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€–
,random_affine_transform_params/zeros/ReshapeReshape5random_affine_transform_params/strided_slice:output:0;random_affine_transform_params/zeros/Reshape/shape:output:0*
T0*
_output_shapes
:o
*random_affine_transform_params/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ∆
$random_affine_transform_params/zerosFill5random_affine_transform_params/zeros/Reshape:output:03random_affine_transform_params/zeros/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€З
4random_affine_transform_params/zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€‘
.random_affine_transform_params/zeros_1/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_1/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ћ
&random_affine_transform_params/zeros_1Fill7random_affine_transform_params/zeros_1/Reshape:output:05random_affine_transform_params/zeros_1/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€Ќ
$random_affine_transform_params/stackPack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/sub_2:z:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/sub_3:z:0-random_affine_transform_params/zeros:output:0/random_affine_transform_params/zeros_1:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

axisЗ
4random_affine_transform_params/zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€‘
.random_affine_transform_params/zeros_2/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_2/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ћ
&random_affine_transform_params/zeros_2Fill7random_affine_transform_params/zeros_2/Reshape:output:05random_affine_transform_params/zeros_2/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€З
4random_affine_transform_params/zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€‘
.random_affine_transform_params/zeros_3/ReshapeReshape5random_affine_transform_params/strided_slice:output:0=random_affine_transform_params/zeros_3/Reshape/shape:output:0*
T0*
_output_shapes
:q
,random_affine_transform_params/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ћ
&random_affine_transform_params/zeros_3Fill7random_affine_transform_params/zeros_3/Reshape:output:05random_affine_transform_params/zeros_3/Const:output:0*
T0*#
_output_shapes
:€€€€€€€€€—
&random_affine_transform_params/stack_1Pack&random_affine_transform_params/Cos:y:0(random_affine_transform_params/Sin_1:y:0(random_affine_transform_params/Neg_1:y:0(random_affine_transform_params/mul_1:z:0(random_affine_transform_params/mul_2:z:0(random_affine_transform_params/Neg_2:y:0/random_affine_transform_params/zeros_2:output:0/random_affine_transform_params/zeros_3:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

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
:€€€€€€€€€цц*
dtype0*
interpolation
BILINEARО
tf.compat.v1.pad/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             µ
tf.compat.v1.pad/PadPad$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.pad/Pad/paddings:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ы
(tf.image.adjust_contrast/adjust_contrastAdjustContrastv2Pimage_projective_transform_layer/ImageProjectiveTransformV3:transformed_images:08tf_image_adjust_contrast_adjust_contrast_contrast_factor*1
_output_shapes
:€€€€€€€€€ццђ
1tf.image.adjust_contrast/adjust_contrast/IdentityIdentity1tf.image.adjust_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццЯ
'ResNet/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ф
ResNet/conv2d_1/Conv2D_1Conv2D:tf.image.adjust_contrast/adjust_contrast/Identity:output:0/ResNet/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
Х
(ResNet/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
ResNet/conv2d_1/BiasAdd_1BiasAdd!ResNet/conv2d_1/Conv2D_1:output:00ResNet/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЯ
-ResNet/batch_normalization_1/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_1/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0п
>ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_1/AssignNewValue*
_output_shapes	
:А*
dtype0х
@ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_1/AssignNewValue_1*
_output_shapes	
:А*
dtype0А
/ResNet/batch_normalization_1/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_1/BiasAdd_1:output:05ResNet/batch_normalization_1/ReadVariableOp_2:value:05ResNet/batch_normalization_1/ReadVariableOp_3:value:0FResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<о
-ResNet/batch_normalization_1/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_1/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_1/AssignNewValue?^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ш
-ResNet/batch_normalization_1/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_1/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_1/AssignNewValue_1A^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(¶
 ResNet/leaky_re_lu_1/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_1/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Ы
%ResNet/conv2d/Conv2D_1/ReadVariableOpReadVariableOp,resnet_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0р
ResNet/conv2d/Conv2D_1Conv2D:tf.image.adjust_contrast/adjust_contrast/Identity:output:0-ResNet/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
С
&ResNet/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp-resnet_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0∞
ResNet/conv2d/BiasAdd_1BiasAddResNet/conv2d/Conv2D_1:output:0.ResNet/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА†
'ResNet/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0и
ResNet/conv2d_2/Conv2D_1Conv2D.ResNet/leaky_re_lu_1/LeakyRelu_1:activations:0/ResNet/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццА*
paddingSAME*
strides
Х
(ResNet/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ґ
ResNet/conv2d_2/BiasAdd_1BiasAdd!ResNet/conv2d_2/Conv2D_1:output:00ResNet/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЫ
+ResNet/batch_normalization/ReadVariableOp_2ReadVariableOp2resnet_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype0Э
+ResNet/batch_normalization/ReadVariableOp_3ReadVariableOp4resnet_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype0й
<ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOpReadVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource*^ResNet/batch_normalization/AssignNewValue*
_output_shapes	
:А*
dtype0п
>ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource,^ResNet/batch_normalization/AssignNewValue_1*
_output_shapes	
:А*
dtype0ф
-ResNet/batch_normalization/FusedBatchNormV3_1FusedBatchNormV3 ResNet/conv2d/BiasAdd_1:output:03ResNet/batch_normalization/ReadVariableOp_2:value:03ResNet/batch_normalization/ReadVariableOp_3:value:0DResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp:value:0FResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<д
+ResNet/batch_normalization/AssignNewValue_2AssignVariableOpCresnet_batch_normalization_fusedbatchnormv3_readvariableop_resource:ResNet/batch_normalization/FusedBatchNormV3_1:batch_mean:0*^ResNet/batch_normalization/AssignNewValue=^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(о
+ResNet/batch_normalization/AssignNewValue_3AssignVariableOpEresnet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource>ResNet/batch_normalization/FusedBatchNormV3_1:batch_variance:0,^ResNet/batch_normalization/AssignNewValue_1?^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Я
-ResNet/batch_normalization_2/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_2/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0п
>ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_2/AssignNewValue*
_output_shapes	
:А*
dtype0х
@ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_2/AssignNewValue_1*
_output_shapes	
:А*
dtype0А
/ResNet/batch_normalization_2/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_2/BiasAdd_1:output:05ResNet/batch_normalization_2/ReadVariableOp_2:value:05ResNet/batch_normalization_2/ReadVariableOp_3:value:0FResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<о
-ResNet/batch_normalization_2/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_2/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_2/AssignNewValue?^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ш
-ResNet/batch_normalization_2/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_2/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_2/AssignNewValue_1A^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(¶
 ResNet/leaky_re_lu_2/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_2/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Ґ
ResNet/leaky_re_lu/LeakyRelu_1	LeakyRelu1ResNet/batch_normalization/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=і
ResNet/add/add_1AddV2.ResNet/leaky_re_lu_2/LeakyRelu_1:activations:0,ResNet/leaky_re_lu/LeakyRelu_1:activations:0*
T0*2
_output_shapes 
:€€€€€€€€€ццАЯ
-ResNet/batch_normalization_3/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0°
-ResNet/batch_normalization_3/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0п
>ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_3/AssignNewValue*
_output_shapes	
:А*
dtype0х
@ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_3/AssignNewValue_1*
_output_shapes	
:А*
dtype0т
/ResNet/batch_normalization_3/FusedBatchNormV3_1FusedBatchNormV3ResNet/add/add_1:z:05ResNet/batch_normalization_3/ReadVariableOp_2:value:05ResNet/batch_normalization_3/ReadVariableOp_3:value:0FResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:€€€€€€€€€ццА:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<о
-ResNet/batch_normalization_3/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_3/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_3/AssignNewValue?^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ш
-ResNet/batch_normalization_3/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_3/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_3/AssignNewValue_1A^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(¶
 ResNet/leaky_re_lu_3/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_3/FusedBatchNormV3_1:y:0*2
_output_shapes 
:€€€€€€€€€ццА*
alpha%Ќћћ=Я
'ResNet/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0з
ResNet/conv2d_4/Conv2D_1Conv2D.ResNet/leaky_re_lu_3/LeakyRelu_1:activations:0/ResNet/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
Ф
(ResNet/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
ResNet/conv2d_4/BiasAdd_1BiasAdd!ResNet/conv2d_4/Conv2D_1:output:00ResNet/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
-ResNet/batch_normalization_5/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_5/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
>ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_5/AssignNewValue*
_output_shapes
:@*
dtype0ф
@ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_5/AssignNewValue_1*
_output_shapes
:@*
dtype0ы
/ResNet/batch_normalization_5/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_4/BiasAdd_1:output:05ResNet/batch_normalization_5/ReadVariableOp_2:value:05ResNet/batch_normalization_5/ReadVariableOp_3:value:0FResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<о
-ResNet/batch_normalization_5/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_5/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_5/AssignNewValue?^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ш
-ResNet/batch_normalization_5/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_5/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_5/AssignNewValue_1A^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(•
 ResNet/leaky_re_lu_5/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_5/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=Я
'ResNet/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0з
ResNet/conv2d_3/Conv2D_1Conv2D.ResNet/leaky_re_lu_3/LeakyRelu_1:activations:0/ResNet/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
Ф
(ResNet/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
ResNet/conv2d_3/BiasAdd_1BiasAdd!ResNet/conv2d_3/Conv2D_1:output:00ResNet/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
'ResNet/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0з
ResNet/conv2d_5/Conv2D_1Conv2D.ResNet/leaky_re_lu_5/LeakyRelu_1:activations:0/ResNet/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@*
paddingSAME*
strides
Ф
(ResNet/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
ResNet/conv2d_5/BiasAdd_1BiasAdd!ResNet/conv2d_5/Conv2D_1:output:00ResNet/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
-ResNet/batch_normalization_4/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_4/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
>ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_4/AssignNewValue*
_output_shapes
:@*
dtype0ф
@ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_4/AssignNewValue_1*
_output_shapes
:@*
dtype0ы
/ResNet/batch_normalization_4/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_3/BiasAdd_1:output:05ResNet/batch_normalization_4/ReadVariableOp_2:value:05ResNet/batch_normalization_4/ReadVariableOp_3:value:0FResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<о
-ResNet/batch_normalization_4/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_4/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_4/AssignNewValue?^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ш
-ResNet/batch_normalization_4/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_4/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_4/AssignNewValue_1A^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ю
-ResNet/batch_normalization_6/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_6/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
>ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_6/AssignNewValue*
_output_shapes
:@*
dtype0ф
@ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_6/AssignNewValue_1*
_output_shapes
:@*
dtype0ы
/ResNet/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3"ResNet/conv2d_5/BiasAdd_1:output:05ResNet/batch_normalization_6/ReadVariableOp_2:value:05ResNet/batch_normalization_6/ReadVariableOp_3:value:0FResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<о
-ResNet/batch_normalization_6/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_6/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_6/AssignNewValue?^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ш
-ResNet/batch_normalization_6/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_6/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_6/AssignNewValue_1A^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(•
 ResNet/leaky_re_lu_6/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_6/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=•
 ResNet/leaky_re_lu_4/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_4/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=Ј
ResNet/add_1/add_1AddV2.ResNet/leaky_re_lu_6/LeakyRelu_1:activations:0.ResNet/leaky_re_lu_4/LeakyRelu_1:activations:0*
T0*1
_output_shapes
:€€€€€€€€€цц@Ю
-ResNet/batch_normalization_7/ReadVariableOp_2ReadVariableOp4resnet_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype0†
-ResNet/batch_normalization_7/ReadVariableOp_3ReadVariableOp6resnet_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
>ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource,^ResNet/batch_normalization_7/AssignNewValue*
_output_shapes
:@*
dtype0ф
@ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource.^ResNet/batch_normalization_7/AssignNewValue_1*
_output_shapes
:@*
dtype0п
/ResNet/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3ResNet/add_1/add_1:z:05ResNet/batch_normalization_7/ReadVariableOp_2:value:05ResNet/batch_normalization_7/ReadVariableOp_3:value:0FResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0HResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€цц@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<о
-ResNet/batch_normalization_7/AssignNewValue_2AssignVariableOpEresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_resource<ResNet/batch_normalization_7/FusedBatchNormV3_1:batch_mean:0,^ResNet/batch_normalization_7/AssignNewValue?^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ш
-ResNet/batch_normalization_7/AssignNewValue_3AssignVariableOpGresnet_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource@ResNet/batch_normalization_7/FusedBatchNormV3_1:batch_variance:0.^ResNet/batch_normalization_7/AssignNewValue_1A^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(•
 ResNet/leaky_re_lu_7/LeakyRelu_1	LeakyRelu3ResNet/batch_normalization_7/FusedBatchNormV3_1:y:0*1
_output_shapes
:€€€€€€€€€цц@*
alpha%Ќћћ=Ю
'ResNet/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp.resnet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0з
ResNet/conv2d_6/Conv2D_1Conv2D.ResNet/leaky_re_lu_7/LeakyRelu_1:activations:0/ResNet/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€цц*
paddingSAME*
strides
Ф
(ResNet/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp/resnet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
ResNet/conv2d_6/BiasAdd_1BiasAdd!ResNet/conv2d_6/Conv2D_1:output:00ResNet/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ццД
ResNet/conv2d_6/Softmax_1Softmax"ResNet/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€ццЫ
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
 *    і
=image_projective_transform_layer_1/ImageProjectiveTransformV3ImageProjectiveTransformV3#ResNet/conv2d_6/Softmax_1:softmax:0-random_affine_transform_params/stack:output:0Simage_projective_transform_layer_1/ImageProjectiveTransformV3/output_shape:output:0Qimage_projective_transform_layer_1/ImageProjectiveTransformV3/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
interpolation
BILINEARА
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             с
"tf.compat.v1.transpose_1/transpose	TransposeRimage_projective_transform_layer_1/ImageProjectiveTransformV3:transformed_images:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*1
_output_shapes
:АА€€€€€€€€€Џ
tf.compat.v1.nn.conv2d/Conv2DConv2Dtf.compat.v1.pad/Pad:output:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*8
_output_shapes&
$:"€€€€€€€€€€€€€€€€€€*
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
valueB"            ч
&tf.__operators__.getitem/strided_sliceStridedSlice&tf.compat.v1.nn.conv2d/Conv2D:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask	*
end_mask	*
shrink_axis_maskБ
tf.compat.v1.squeeze/SqueezeSqueeze/tf.__operators__.getitem/strided_slice:output:0*
T0*
_output_shapes

:^
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А@Ц
tf.math.truediv/truedivRealDiv%tf.compat.v1.squeeze/Squeeze:output:0"tf.math.truediv/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCР
tf.math.truediv_1/truedivRealDivtf.math.truediv/truediv:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*
_output_shapes

:`
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  АCТ
tf.math.truediv_2/truedivRealDivtf.math.truediv_1/truediv:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*
_output_shapes

:x
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ©
"tf.compat.v1.transpose_2/transpose	Transposetf.math.truediv_2/truediv:z:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*
_output_shapes

:У
tf.__operators__.add/AddV2AddV2tf.math.truediv_2/truediv:z:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*
_output_shapes

:`
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
tf.math.truediv_3/truedivRealDivtf.__operators__.add/AddV2:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Р
tf.__operators__.add_2/AddV2AddV2tf.math.truediv_3/truediv:z:0!tf.__operators__.add_2/y:output:0*
T0*
_output_shapes

:s
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€©
tf.math.reduce_sum/SumSumtf.math.truediv_3/truediv:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(u
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€≠
tf.math.reduce_sum_1/SumSumtf.math.truediv_3/truediv:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(И
tf.math.multiply/MulMultf.math.reduce_sum/Sum:output:0!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes

:]
tf.__operators__.add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Хњ÷3Л
tf.__operators__.add_1/AddV2AddV2tf.math.multiply/Mul:z:0!tf.__operators__.add_1/y:output:0*
T0*
_output_shapes

:С
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_1/AddV2:z:0 tf.__operators__.add_2/AddV2:z:0*
T0*
_output_shapes

:^
tf.math.log/LogLogtf.math.truediv_4/truediv:z:0*
T0*
_output_shapes

:z
tf.math.multiply_1/MulMultf.math.truediv_3/truediv:z:0tf.math.log/Log:y:0*
T0*
_output_shapes

:{
*tf.math.reduce_sum_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"€€€€ю€€€С
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
value	B :≥
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€a

Identity_1Identity!tf.math.reduce_mean/Mean:output:0^NoOp*
T0*
_output_shapes
: ∞1
NoOpNoOp*^ResNet/batch_normalization/AssignNewValue,^ResNet/batch_normalization/AssignNewValue_1,^ResNet/batch_normalization/AssignNewValue_2,^ResNet/batch_normalization/AssignNewValue_3;^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp=^ResNet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp?^ResNet/batch_normalization/FusedBatchNormV3_1/ReadVariableOp_1*^ResNet/batch_normalization/ReadVariableOp,^ResNet/batch_normalization/ReadVariableOp_1,^ResNet/batch_normalization/ReadVariableOp_2,^ResNet/batch_normalization/ReadVariableOp_3,^ResNet/batch_normalization_1/AssignNewValue.^ResNet/batch_normalization_1/AssignNewValue_1.^ResNet/batch_normalization_1/AssignNewValue_2.^ResNet/batch_normalization_1/AssignNewValue_3=^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_1/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_1/ReadVariableOp.^ResNet/batch_normalization_1/ReadVariableOp_1.^ResNet/batch_normalization_1/ReadVariableOp_2.^ResNet/batch_normalization_1/ReadVariableOp_3,^ResNet/batch_normalization_2/AssignNewValue.^ResNet/batch_normalization_2/AssignNewValue_1.^ResNet/batch_normalization_2/AssignNewValue_2.^ResNet/batch_normalization_2/AssignNewValue_3=^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_2/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_2/ReadVariableOp.^ResNet/batch_normalization_2/ReadVariableOp_1.^ResNet/batch_normalization_2/ReadVariableOp_2.^ResNet/batch_normalization_2/ReadVariableOp_3,^ResNet/batch_normalization_3/AssignNewValue.^ResNet/batch_normalization_3/AssignNewValue_1.^ResNet/batch_normalization_3/AssignNewValue_2.^ResNet/batch_normalization_3/AssignNewValue_3=^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_3/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_3/ReadVariableOp.^ResNet/batch_normalization_3/ReadVariableOp_1.^ResNet/batch_normalization_3/ReadVariableOp_2.^ResNet/batch_normalization_3/ReadVariableOp_3,^ResNet/batch_normalization_4/AssignNewValue.^ResNet/batch_normalization_4/AssignNewValue_1.^ResNet/batch_normalization_4/AssignNewValue_2.^ResNet/batch_normalization_4/AssignNewValue_3=^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_4/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_4/ReadVariableOp.^ResNet/batch_normalization_4/ReadVariableOp_1.^ResNet/batch_normalization_4/ReadVariableOp_2.^ResNet/batch_normalization_4/ReadVariableOp_3,^ResNet/batch_normalization_5/AssignNewValue.^ResNet/batch_normalization_5/AssignNewValue_1.^ResNet/batch_normalization_5/AssignNewValue_2.^ResNet/batch_normalization_5/AssignNewValue_3=^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_5/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_5/ReadVariableOp.^ResNet/batch_normalization_5/ReadVariableOp_1.^ResNet/batch_normalization_5/ReadVariableOp_2.^ResNet/batch_normalization_5/ReadVariableOp_3,^ResNet/batch_normalization_6/AssignNewValue.^ResNet/batch_normalization_6/AssignNewValue_1.^ResNet/batch_normalization_6/AssignNewValue_2.^ResNet/batch_normalization_6/AssignNewValue_3=^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_6/ReadVariableOp.^ResNet/batch_normalization_6/ReadVariableOp_1.^ResNet/batch_normalization_6/ReadVariableOp_2.^ResNet/batch_normalization_6/ReadVariableOp_3,^ResNet/batch_normalization_7/AssignNewValue.^ResNet/batch_normalization_7/AssignNewValue_1.^ResNet/batch_normalization_7/AssignNewValue_2.^ResNet/batch_normalization_7/AssignNewValue_3=^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?^ResNet/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpA^ResNet/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1,^ResNet/batch_normalization_7/ReadVariableOp.^ResNet/batch_normalization_7/ReadVariableOp_1.^ResNet/batch_normalization_7/ReadVariableOp_2.^ResNet/batch_normalization_7/ReadVariableOp_3%^ResNet/conv2d/BiasAdd/ReadVariableOp'^ResNet/conv2d/BiasAdd_1/ReadVariableOp$^ResNet/conv2d/Conv2D/ReadVariableOp&^ResNet/conv2d/Conv2D_1/ReadVariableOp'^ResNet/conv2d_1/BiasAdd/ReadVariableOp)^ResNet/conv2d_1/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_1/Conv2D/ReadVariableOp(^ResNet/conv2d_1/Conv2D_1/ReadVariableOp'^ResNet/conv2d_2/BiasAdd/ReadVariableOp)^ResNet/conv2d_2/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_2/Conv2D/ReadVariableOp(^ResNet/conv2d_2/Conv2D_1/ReadVariableOp'^ResNet/conv2d_3/BiasAdd/ReadVariableOp)^ResNet/conv2d_3/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_3/Conv2D/ReadVariableOp(^ResNet/conv2d_3/Conv2D_1/ReadVariableOp'^ResNet/conv2d_4/BiasAdd/ReadVariableOp)^ResNet/conv2d_4/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_4/Conv2D/ReadVariableOp(^ResNet/conv2d_4/Conv2D_1/ReadVariableOp'^ResNet/conv2d_5/BiasAdd/ReadVariableOp)^ResNet/conv2d_5/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_5/Conv2D/ReadVariableOp(^ResNet/conv2d_5/Conv2D_1/ReadVariableOp'^ResNet/conv2d_6/BiasAdd/ReadVariableOp)^ResNet/conv2d_6/BiasAdd_1/ReadVariableOp&^ResNet/conv2d_6/Conv2D/ReadVariableOp(^ResNet/conv2d_6/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:/

_output_shapes
: 
 
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_36534

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Зї
чK
!__inference__traced_restore_37526
file_prefix;
 assignvariableop_conv2d_1_kernel:А/
 assignvariableop_1_conv2d_1_bias:	А=
.assignvariableop_2_batch_normalization_1_gamma:	А<
-assignvariableop_3_batch_normalization_1_beta:	АC
4assignvariableop_4_batch_normalization_1_moving_mean:	АG
8assignvariableop_5_batch_normalization_1_moving_variance:	А>
"assignvariableop_6_conv2d_2_kernel:АА/
 assignvariableop_7_conv2d_2_bias:	А;
 assignvariableop_8_conv2d_kernel:А-
assignvariableop_9_conv2d_bias:	А>
/assignvariableop_10_batch_normalization_2_gamma:	А=
.assignvariableop_11_batch_normalization_2_beta:	АD
5assignvariableop_12_batch_normalization_2_moving_mean:	АH
9assignvariableop_13_batch_normalization_2_moving_variance:	А<
-assignvariableop_14_batch_normalization_gamma:	А;
,assignvariableop_15_batch_normalization_beta:	АB
3assignvariableop_16_batch_normalization_moving_mean:	АF
7assignvariableop_17_batch_normalization_moving_variance:	А>
/assignvariableop_18_batch_normalization_3_gamma:	А=
.assignvariableop_19_batch_normalization_3_beta:	АD
5assignvariableop_20_batch_normalization_3_moving_mean:	АH
9assignvariableop_21_batch_normalization_3_moving_variance:	А>
#assignvariableop_22_conv2d_4_kernel:А@/
!assignvariableop_23_conv2d_4_bias:@=
/assignvariableop_24_batch_normalization_5_gamma:@<
.assignvariableop_25_batch_normalization_5_beta:@C
5assignvariableop_26_batch_normalization_5_moving_mean:@G
9assignvariableop_27_batch_normalization_5_moving_variance:@=
#assignvariableop_28_conv2d_5_kernel:@@/
!assignvariableop_29_conv2d_5_bias:@>
#assignvariableop_30_conv2d_3_kernel:А@/
!assignvariableop_31_conv2d_3_bias:@=
/assignvariableop_32_batch_normalization_6_gamma:@<
.assignvariableop_33_batch_normalization_6_beta:@C
5assignvariableop_34_batch_normalization_6_moving_mean:@G
9assignvariableop_35_batch_normalization_6_moving_variance:@=
/assignvariableop_36_batch_normalization_4_gamma:@<
.assignvariableop_37_batch_normalization_4_beta:@C
5assignvariableop_38_batch_normalization_4_moving_mean:@G
9assignvariableop_39_batch_normalization_4_moving_variance:@=
/assignvariableop_40_batch_normalization_7_gamma:@<
.assignvariableop_41_batch_normalization_7_beta:@C
5assignvariableop_42_batch_normalization_7_moving_mean:@G
9assignvariableop_43_batch_normalization_7_moving_variance:@=
#assignvariableop_44_conv2d_6_kernel:@/
!assignvariableop_45_conv2d_6_bias:'
assignvariableop_46_adam_iter:	 )
assignvariableop_47_adam_beta_1: )
assignvariableop_48_adam_beta_2: (
assignvariableop_49_adam_decay: #
assignvariableop_50_total: #
assignvariableop_51_count: E
*assignvariableop_52_adam_conv2d_1_kernel_m:А7
(assignvariableop_53_adam_conv2d_1_bias_m:	АE
6assignvariableop_54_adam_batch_normalization_1_gamma_m:	АD
5assignvariableop_55_adam_batch_normalization_1_beta_m:	АF
*assignvariableop_56_adam_conv2d_2_kernel_m:АА7
(assignvariableop_57_adam_conv2d_2_bias_m:	АC
(assignvariableop_58_adam_conv2d_kernel_m:А5
&assignvariableop_59_adam_conv2d_bias_m:	АE
6assignvariableop_60_adam_batch_normalization_2_gamma_m:	АD
5assignvariableop_61_adam_batch_normalization_2_beta_m:	АC
4assignvariableop_62_adam_batch_normalization_gamma_m:	АB
3assignvariableop_63_adam_batch_normalization_beta_m:	АE
6assignvariableop_64_adam_batch_normalization_3_gamma_m:	АD
5assignvariableop_65_adam_batch_normalization_3_beta_m:	АE
*assignvariableop_66_adam_conv2d_4_kernel_m:А@6
(assignvariableop_67_adam_conv2d_4_bias_m:@D
6assignvariableop_68_adam_batch_normalization_5_gamma_m:@C
5assignvariableop_69_adam_batch_normalization_5_beta_m:@D
*assignvariableop_70_adam_conv2d_5_kernel_m:@@6
(assignvariableop_71_adam_conv2d_5_bias_m:@E
*assignvariableop_72_adam_conv2d_3_kernel_m:А@6
(assignvariableop_73_adam_conv2d_3_bias_m:@D
6assignvariableop_74_adam_batch_normalization_6_gamma_m:@C
5assignvariableop_75_adam_batch_normalization_6_beta_m:@D
6assignvariableop_76_adam_batch_normalization_4_gamma_m:@C
5assignvariableop_77_adam_batch_normalization_4_beta_m:@D
6assignvariableop_78_adam_batch_normalization_7_gamma_m:@C
5assignvariableop_79_adam_batch_normalization_7_beta_m:@D
*assignvariableop_80_adam_conv2d_6_kernel_m:@6
(assignvariableop_81_adam_conv2d_6_bias_m:E
*assignvariableop_82_adam_conv2d_1_kernel_v:А7
(assignvariableop_83_adam_conv2d_1_bias_v:	АE
6assignvariableop_84_adam_batch_normalization_1_gamma_v:	АD
5assignvariableop_85_adam_batch_normalization_1_beta_v:	АF
*assignvariableop_86_adam_conv2d_2_kernel_v:АА7
(assignvariableop_87_adam_conv2d_2_bias_v:	АC
(assignvariableop_88_adam_conv2d_kernel_v:А5
&assignvariableop_89_adam_conv2d_bias_v:	АE
6assignvariableop_90_adam_batch_normalization_2_gamma_v:	АD
5assignvariableop_91_adam_batch_normalization_2_beta_v:	АC
4assignvariableop_92_adam_batch_normalization_gamma_v:	АB
3assignvariableop_93_adam_batch_normalization_beta_v:	АE
6assignvariableop_94_adam_batch_normalization_3_gamma_v:	АD
5assignvariableop_95_adam_batch_normalization_3_beta_v:	АE
*assignvariableop_96_adam_conv2d_4_kernel_v:А@6
(assignvariableop_97_adam_conv2d_4_bias_v:@D
6assignvariableop_98_adam_batch_normalization_5_gamma_v:@C
5assignvariableop_99_adam_batch_normalization_5_beta_v:@E
+assignvariableop_100_adam_conv2d_5_kernel_v:@@7
)assignvariableop_101_adam_conv2d_5_bias_v:@F
+assignvariableop_102_adam_conv2d_3_kernel_v:А@7
)assignvariableop_103_adam_conv2d_3_bias_v:@E
7assignvariableop_104_adam_batch_normalization_6_gamma_v:@D
6assignvariableop_105_adam_batch_normalization_6_beta_v:@E
7assignvariableop_106_adam_batch_normalization_4_gamma_v:@D
6assignvariableop_107_adam_batch_normalization_4_beta_v:@E
7assignvariableop_108_adam_batch_normalization_7_gamma_v:@D
6assignvariableop_109_adam_batch_normalization_7_beta_v:@E
+assignvariableop_110_adam_conv2d_6_kernel_v:@7
)assignvariableop_111_adam_conv2d_6_bias_v:
identity_113ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_102ҐAssignVariableOp_103ҐAssignVariableOp_104ҐAssignVariableOp_105ҐAssignVariableOp_106ҐAssignVariableOp_107ҐAssignVariableOp_108ҐAssignVariableOp_109ҐAssignVariableOp_11ҐAssignVariableOp_110ҐAssignVariableOp_111ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99Ъ2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:q*
dtype0*ј1
valueґ1B≥1qB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH’
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:q*
dtype0*ч
valueнBкqB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ÷
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Џ
_output_shapes«
ƒ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypesu
s2q	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv2d_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_2_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_2_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_14AssignVariableOp-assignvariableop_14_batch_normalization_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_15AssignVariableOp,assignvariableop_15_batch_normalization_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_16AssignVariableOp3assignvariableop_16_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_17AssignVariableOp7assignvariableop_17_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_3_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_3_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_5_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_5_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_5_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_5_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_28AssignVariableOp#assignvariableop_28_conv2d_5_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_29AssignVariableOp!assignvariableop_29_conv2d_5_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_6_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_6_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_6_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_6_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_4_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_4_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_4_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_4_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_40AssignVariableOp/assignvariableop_40_batch_normalization_7_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_41AssignVariableOp.assignvariableop_41_batch_normalization_7_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_42AssignVariableOp5assignvariableop_42_batch_normalization_7_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_43AssignVariableOp9assignvariableop_43_batch_normalization_7_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_44AssignVariableOp#assignvariableop_44_conv2d_6_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_45AssignVariableOp!assignvariableop_45_conv2d_6_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_46AssignVariableOpassignvariableop_46_adam_iterIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_beta_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_beta_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_decayIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_50AssignVariableOpassignvariableop_50_totalIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_51AssignVariableOpassignvariableop_51_countIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_conv2d_1_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv2d_1_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_1_gamma_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_batch_normalization_1_beta_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_2_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_conv2d_2_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv2d_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_conv2d_bias_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_2_gamma_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adam_batch_normalization_2_beta_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adam_batch_normalization_gamma_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_63AssignVariableOp3assignvariableop_63_adam_batch_normalization_beta_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_3_gamma_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_65AssignVariableOp5assignvariableop_65_adam_batch_normalization_3_beta_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_4_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_conv2d_4_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_5_gamma_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adam_batch_normalization_5_beta_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_conv2d_5_kernel_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_conv2d_5_bias_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_conv2d_3_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_conv2d_3_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_batch_normalization_6_gamma_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_75AssignVariableOp5assignvariableop_75_adam_batch_normalization_6_beta_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_4_gamma_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_77AssignVariableOp5assignvariableop_77_adam_batch_normalization_4_beta_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_batch_normalization_7_gamma_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_79AssignVariableOp5assignvariableop_79_adam_batch_normalization_7_beta_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_conv2d_6_kernel_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adam_conv2d_6_bias_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_conv2d_1_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_conv2d_1_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_1_gamma_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_85AssignVariableOp5assignvariableop_85_adam_batch_normalization_1_beta_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_conv2d_2_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_conv2d_2_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv2d_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_89AssignVariableOp&assignvariableop_89_adam_conv2d_bias_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_batch_normalization_2_gamma_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_91AssignVariableOp5assignvariableop_91_adam_batch_normalization_2_beta_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_92AssignVariableOp4assignvariableop_92_adam_batch_normalization_gamma_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_93AssignVariableOp3assignvariableop_93_adam_batch_normalization_beta_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_94AssignVariableOp6assignvariableop_94_adam_batch_normalization_3_gamma_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_95AssignVariableOp5assignvariableop_95_adam_batch_normalization_3_beta_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_conv2d_4_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_97AssignVariableOp(assignvariableop_97_adam_conv2d_4_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_98AssignVariableOp6assignvariableop_98_adam_batch_normalization_5_gamma_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_99AssignVariableOp5assignvariableop_99_adam_batch_normalization_5_beta_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_100AssignVariableOp+assignvariableop_100_adam_conv2d_5_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_101AssignVariableOp)assignvariableop_101_adam_conv2d_5_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_102AssignVariableOp+assignvariableop_102_adam_conv2d_3_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_103AssignVariableOp)assignvariableop_103_adam_conv2d_3_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_104AssignVariableOp7assignvariableop_104_adam_batch_normalization_6_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_105AssignVariableOp6assignvariableop_105_adam_batch_normalization_6_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_106AssignVariableOp7assignvariableop_106_adam_batch_normalization_4_gamma_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_107AssignVariableOp6assignvariableop_107_adam_batch_normalization_4_beta_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_batch_normalization_7_gamma_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_109AssignVariableOp6assignvariableop_109_adam_batch_normalization_7_beta_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_110AssignVariableOp+assignvariableop_110_adam_conv2d_6_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_111AssignVariableOp)assignvariableop_111_adam_conv2d_6_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ь
Identity_112Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_113IdentityIdentity_112:output:0^NoOp_1*
T0*
_output_shapes
: и
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_113Identity_113:output:0*ч
_input_shapesе
в: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112*
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
џ
Я
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31329

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
џ
Я
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36259

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ћ
Ы
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36506

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
М
€
C__inference_conv2d_2_layer_call_and_return_conditional_losses_36196

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Е
њ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36790

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ќ
d
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_31944

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
 
d
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_36706

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
э
ы

%__inference_model_layer_call_fn_33370
input_1"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45
identityИҐStatefulPartitionedCallд
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
unknown_44
unknown_45*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: *P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_33272Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:/

_output_shapes
: 
Ф	
–
5__inference_batch_normalization_4_layer_call_fn_36647

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31713Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
 
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_31976

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Љ
л

&__inference_ResNet_layer_call_fn_35523

inputs"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:
identityИҐStatefulPartitionedCall‘
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_ResNet_layer_call_and_return_conditional_losses_32084Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ю
Ј
A__inference_ResNet_layer_call_and_return_conditional_losses_32965
input_2)
conv2d_1_32847:А
conv2d_1_32849:	А*
batch_normalization_1_32852:	А*
batch_normalization_1_32854:	А*
batch_normalization_1_32856:	А*
batch_normalization_1_32858:	А'
conv2d_32862:А
conv2d_32864:	А*
conv2d_2_32867:АА
conv2d_2_32869:	А(
batch_normalization_32872:	А(
batch_normalization_32874:	А(
batch_normalization_32876:	А(
batch_normalization_32878:	А*
batch_normalization_2_32881:	А*
batch_normalization_2_32883:	А*
batch_normalization_2_32885:	А*
batch_normalization_2_32887:	А*
batch_normalization_3_32893:	А*
batch_normalization_3_32895:	А*
batch_normalization_3_32897:	А*
batch_normalization_3_32899:	А)
conv2d_4_32903:А@
conv2d_4_32905:@)
batch_normalization_5_32908:@)
batch_normalization_5_32910:@)
batch_normalization_5_32912:@)
batch_normalization_5_32914:@)
conv2d_3_32918:А@
conv2d_3_32920:@(
conv2d_5_32923:@@
conv2d_5_32925:@)
batch_normalization_4_32928:@)
batch_normalization_4_32930:@)
batch_normalization_4_32932:@)
batch_normalization_4_32934:@)
batch_normalization_6_32937:@)
batch_normalization_6_32939:@)
batch_normalization_6_32941:@)
batch_normalization_6_32943:@)
batch_normalization_7_32949:@)
batch_normalization_7_32951:@)
batch_normalization_7_32953:@)
batch_normalization_7_32955:@(
conv2d_6_32959:@
conv2d_6_32961:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallО
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_1_32847conv2d_1_32849*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_31836†
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_32852batch_normalization_1_32854batch_normalization_1_32856batch_normalization_1_32858*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31360С
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_31856Ж
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_32862conv2d_32864*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_31868≠
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv2d_2_32867conv2d_2_32869*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_31884Т
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_32872batch_normalization_32874batch_normalization_32876batch_normalization_32878*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31488†
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_32881batch_normalization_2_32883batch_normalization_2_32885batch_normalization_2_32887*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31424С
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_31913Л
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_31920Ф
add/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_31928У
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_32893batch_normalization_3_32895batch_normalization_3_32897batch_normalization_3_32899*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31552С
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_31944ђ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_4_32903conv2d_4_32905*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_31956Я
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_32908batch_normalization_5_32910batch_normalization_5_32912batch_normalization_5_32914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31616Р
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_31976ђ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_3_32918conv2d_3_32920*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_31988ђ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0conv2d_5_32923conv2d_5_32925*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_32004Я
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_32928batch_normalization_4_32930batch_normalization_4_32932batch_normalization_4_32934*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31744Я
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_32937batch_normalization_6_32939batch_normalization_6_32941batch_normalization_6_32943*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31680Р
leaky_re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_32033Р
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_32040Щ
add_1/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_32048Ф
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_32949batch_normalization_7_32951batch_normalization_7_32953batch_normalization_7_32955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31808Р
leaky_re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_32064ђ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv2d_6_32959conv2d_6_32961*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_32077Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_2
 
d
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_32064

inputs
identityq
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
alpha%Ќћћ=y
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ќ
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_36177

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ж
ь
A__inference_conv2d_layer_call_and_return_conditional_losses_31868

inputs9
conv2d_readvariableop_resource:А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ў
Э
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31457

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Е
њ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36696

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Х
√
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36167

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ў
щ

#__inference_signature_wrapper_34392
input_1"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А$
	unknown_5:А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А%

unknown_21:А@

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:А@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45
identityИҐStatefulPartitionedCallЅ
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
unknown_44
unknown_45*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*2
config_proto" 

CPU

GPU2 *0J 8В *)
f$R"
 __inference__wrapped_model_31307Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*†
_input_shapesО
Л:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
!
_user_specified_name	input_1:/

_output_shapes
: 
љ
†
(__inference_conv2d_2_layer_call_fn_36186

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_31884К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Е
э
C__inference_conv2d_4_layer_call_and_return_conditional_losses_31956

inputs9
conv2d_readvariableop_resource:А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Е
њ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31808

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ц
I
-__inference_leaky_re_lu_1_layer_call_fn_36172

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_31856{
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ЛE
Е
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_36047
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
valueB:—
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
:€€€€€€€€€*
dtype0c
RoundRound%random_uniform/RandomUniform:output:0*
T0*#
_output_shapes
:€€€€€€€€€J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @S
mulMul	Round:y:0mul/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Q
subSubmul:z:0sub/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€d
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
 *џIјY
random_uniform_1/maxConst*
_output_shapes
: *
dtype0*
valueB
 *џI@Л
random_uniform_1/RandomUniformRandomUniformrandom_uniform_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype0z
random_uniform_1/subSubrandom_uniform_1/max:output:0random_uniform_1/min:output:0*
T0*
_output_shapes
: М
random_uniform_1/mulMul'random_uniform_1/RandomUniform:output:0random_uniform_1/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€А
random_uniform_1AddV2random_uniform_1/mul:z:0random_uniform_1/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€N
CosCosrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€N
SinSinrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€A
NegNegSin:y:0*
T0*#
_output_shapes
:€€€€€€€€€L
mul_1MulNeg:y:0sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
Sin_1Sinrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
Cos_1Cosrandom_uniform_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€N
mul_2Mul	Cos_1:y:0sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€_
packed/0PackCos:y:0	mul_1:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€a
packed/1Pack	Sin_1:y:0	mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€s
packedPackpacked/0:output:0packed/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€a

packed_1/0PackCos:y:0	Sin_1:y:0*
N*
T0*'
_output_shapes
:€€€€€€€€€c

packed_1/1Pack	mul_1:z:0	mul_2:z:0*
N*
T0*'
_output_shapes
:€€€€€€€€€y
packed_1Packpacked_1/0:output:0packed_1/1:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_1	Transposepacked_1:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
ConstConst*
_output_shapes

:*
dtype0*!
valueB"у5Cу5C`
Const_1Const*
_output_shapes

:*
dtype0*!
valueB"   C   Cn
MatMulBatchMatMulV2transpose:y:0Const_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€c
sub_1SubConst:output:0MatMul:output:0*
T0*+
_output_shapes
:€€€€€€€€€k
MatMul_1BatchMatMulV2transpose_1:y:0	sub_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
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
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskT
Neg_1Negstrided_slice_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€j
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
:€€€€€€€€€*

begin_mask*
end_mask*
shrink_axis_maskT
Neg_2Negstrided_slice_2:output:0*
T0*#
_output_shapes
:€€€€€€€€€E
Neg_3Neg	Neg_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€N
mul_3Mul	Neg_3:y:0Cos:y:0*
T0*#
_output_shapes
:€€€€€€€€€P
mul_4Mul	Neg_2:y:0	mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
sub_2Sub	mul_3:z:0	mul_4:z:0*
T0*#
_output_shapes
:€€€€€€€€€E
Neg_4Neg	Neg_2:y:0*
T0*#
_output_shapes
:€€€€€€€€€P
mul_5Mul	Neg_4:y:0	mul_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€P
mul_6Mul	Neg_1:y:0	Sin_1:y:0*
T0*#
_output_shapes
:€€€€€€€€€P
sub_3Sub	mul_5:z:0	mul_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€f
zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€s
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
:€€€€€€€€€h
zeros_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
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
:€€€€€€€€€ґ
stackPackCos:y:0	mul_1:z:0	sub_2:z:0	Sin_1:y:0	mul_2:z:0	sub_3:z:0zeros:output:0zeros_1:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

axish
zeros_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
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
:€€€€€€€€€h
zeros_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
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
:€€€€€€€€€Ї
stack_1PackCos:y:0	Sin_1:y:0	Neg_1:y:0	mul_1:z:0	mul_2:z:0	Neg_2:y:0zeros_2:output:0zeros_3:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€*

axisX
IdentityIdentitystack_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€X

Identity_1Identitystack:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:f b
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

_user_specified_nameinp
Ж
ь
A__inference_conv2d_layer_call_and_return_conditional_losses_36215

inputs9
conv2d_readvariableop_resource:А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Е
њ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36634

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Е
њ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31744

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Т	
–
5__inference_batch_normalization_7_layer_call_fn_36754

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31808Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
я
D
(__inference_add_loss_layer_call_fn_36081

inputs
identity•
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
GPU2 *0J 8В *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_33267O
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
Ё
j
>__inference_add_layer_call_and_return_conditional_losses_36371
inputs_0
inputs_1
identitym
addAddV2inputs_0inputs_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аj
IdentityIdentityadd:z:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:l h
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/1
є
Ю
(__inference_conv2d_3_layer_call_fn_36562

inputs"
unknown:А@
	unknown_0:@
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_31988Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Е
э
C__inference_conv2d_3_layer_call_and_return_conditional_losses_31988

inputs9
conv2d_readvariableop_resource:А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
М
€
C__inference_conv2d_2_layer_call_and_return_conditional_losses_31884

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Р
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
КА
ґ
A__inference_ResNet_layer_call_and_return_conditional_losses_32084

inputs)
conv2d_1_31837:А
conv2d_1_31839:	А*
batch_normalization_1_31842:	А*
batch_normalization_1_31844:	А*
batch_normalization_1_31846:	А*
batch_normalization_1_31848:	А'
conv2d_31869:А
conv2d_31871:	А*
conv2d_2_31885:АА
conv2d_2_31887:	А(
batch_normalization_31890:	А(
batch_normalization_31892:	А(
batch_normalization_31894:	А(
batch_normalization_31896:	А*
batch_normalization_2_31899:	А*
batch_normalization_2_31901:	А*
batch_normalization_2_31903:	А*
batch_normalization_2_31905:	А*
batch_normalization_3_31930:	А*
batch_normalization_3_31932:	А*
batch_normalization_3_31934:	А*
batch_normalization_3_31936:	А)
conv2d_4_31957:А@
conv2d_4_31959:@)
batch_normalization_5_31962:@)
batch_normalization_5_31964:@)
batch_normalization_5_31966:@)
batch_normalization_5_31968:@)
conv2d_3_31989:А@
conv2d_3_31991:@(
conv2d_5_32005:@@
conv2d_5_32007:@)
batch_normalization_4_32010:@)
batch_normalization_4_32012:@)
batch_normalization_4_32014:@)
batch_normalization_4_32016:@)
batch_normalization_6_32019:@)
batch_normalization_6_32021:@)
batch_normalization_6_32023:@)
batch_normalization_6_32025:@)
batch_normalization_7_32050:@)
batch_normalization_7_32052:@)
batch_normalization_7_32054:@)
batch_normalization_7_32056:@(
conv2d_6_32078:@
conv2d_6_32080:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallН
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_31837conv2d_1_31839*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_31836Ґ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_31842batch_normalization_1_31844batch_normalization_1_31846batch_normalization_1_31848*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_31329С
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_31856Е
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_31869conv2d_31871*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_31868≠
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0conv2d_2_31885conv2d_2_31887*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_31884Ф
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_31890batch_normalization_31892batch_normalization_31894batch_normalization_31896*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31457Ґ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_31899batch_normalization_2_31901batch_normalization_2_31903batch_normalization_2_31905*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31393С
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_31913Л
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_31920Ф
add/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_31928Х
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_31930batch_normalization_3_31932batch_normalization_3_31934batch_normalization_3_31936*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31521С
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_31944ђ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_4_31957conv2d_4_31959*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_31956°
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_5_31962batch_normalization_5_31964batch_normalization_5_31966batch_normalization_5_31968*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31585Р
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_31976ђ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0conv2d_3_31989conv2d_3_31991*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_31988ђ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0conv2d_5_32005conv2d_5_32007*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_32004°
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_32010batch_normalization_4_32012batch_normalization_4_32014batch_normalization_4_32016*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_31713°
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_6_32019batch_normalization_6_32021batch_normalization_6_32023batch_normalization_6_32025*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31649Р
leaky_re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_32033Р
leaky_re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_32040Щ
add_1/PartitionedCallPartitionedCall&leaky_re_lu_6/PartitionedCall:output:0&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_32048Ц
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_7_32050batch_normalization_7_32052batch_normalization_7_32054batch_normalization_7_32056*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31777Р
leaky_re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_32064ђ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_7/PartitionedCall:output:0conv2d_6_32078conv2d_6_32080*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_32077Т
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ј
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ю
_input_shapesМ
Й:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ь	
‘
5__inference_batch_normalization_3_layer_call_fn_36384

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_31521К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ћ
Ы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_31649

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ц	
“
3__inference_batch_normalization_layer_call_fn_36303

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_31488К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
—
j
@__inference_add_1_layer_call_and_return_conditional_losses_32048

inputs
inputs_1
identityj
addAddV2inputsinputs_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@i
IdentityIdentityadd:z:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ћ
Ы
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36678

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ї
Я
(__inference_conv2d_1_layer_call_fn_36095

inputs"
unknown:А
	unknown_0:	А
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_31836К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ф	
–
5__inference_batch_normalization_7_layer_call_fn_36741

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_31777Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ћ
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_31920

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
џ
Я
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36149

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
џ
Я
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_31393

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ћ
Ы
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_31585

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Б
ь
C__inference_conv2d_5_layer_call_and_return_conditional_losses_36553

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ђ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Х
√
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36277

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ћ
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_36359

inputs
identityr
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
alpha%Ќћћ=z
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs"њL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ё
serving_default…
U
input_1J
serving_default_input_1:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€T
ResNetJ
StatefulPartitionedCall:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€tensorflow/serving/predict:ґЪ
У
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
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&loss
'
signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ф
(layer-0
)layer_with_weights-0
)layer-1
*layer_with_weights-1
*layer-2
+layer-3
,layer_with_weights-2
,layer-4
-layer_with_weights-3
-layer-5
.layer_with_weights-4
.layer-6
/layer_with_weights-5
/layer-7
0layer-8
1layer-9
2layer-10
3layer_with_weights-6
3layer-11
4layer-12
5layer_with_weights-7
5layer-13
6layer_with_weights-8
6layer-14
7layer-15
8layer_with_weights-9
8layer-16
9layer_with_weights-10
9layer-17
:layer_with_weights-11
:layer-18
;layer_with_weights-12
;layer-19
<layer-20
=layer-21
>layer-22
?layer_with_weights-13
?layer-23
@layer-24
Alayer_with_weights-14
Alayer-25
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_network
•
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
•
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
(
T	keras_api"
_tf_keras_layer
(
U	keras_api"
_tf_keras_layer
•
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
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
(
o	keras_api"
_tf_keras_layer
(
p	keras_api"
_tf_keras_layer
•
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
w0
x1
y2
z3
{4
|5
}6
~7
8
А9
Б10
В11
Г12
Д13
Е14
Ж15
З16
И17
Й18
К19
Л20
М21
Н22
О23
П24
Р25
С26
Т27
У28
Ф29
Х30
Ц31
Ч32
Ш33
Щ34
Ъ35
Ы36
Ь37
Э38
Ю39
Я40
†41
°42
Ґ43
£44
§45"
trackable_list_wrapper
Э
w0
x1
y2
z3
}4
~5
6
А7
Б8
В9
Е10
Ж11
Й12
К13
Н14
О15
П16
Р17
У18
Ф19
Х20
Ц21
Ч22
Ш23
Ы24
Ь25
Я26
†27
£28
§29"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
“
™trace_0
Ђtrace_1
ђtrace_2
≠trace_32я
%__inference_model_layer_call_fn_33370
%__inference_model_layer_call_fn_34492
%__inference_model_layer_call_fn_34592
%__inference_model_layer_call_fn_33895ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z™trace_0zЂtrace_1zђtrace_2z≠trace_3
Њ
Ѓtrace_0
ѓtrace_1
∞trace_2
±trace_32Ћ
@__inference_model_layer_call_and_return_conditional_losses_35009
@__inference_model_layer_call_and_return_conditional_losses_35426
@__inference_model_layer_call_and_return_conditional_losses_34091
@__inference_model_layer_call_and_return_conditional_losses_34287ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zЃtrace_0zѓtrace_1z∞trace_2z±trace_3
ЋB»
 __inference__wrapped_model_31307input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 
	≤iter
≥beta_1
іbeta_2

µdecaywm…xm ymЋzmћ}mЌ~mќmѕ	Аm–	Бm—	Вm“	Еm”	Жm‘	Йm’	Кm÷	Нm„	ОmЎ	Пmў	РmЏ	Уmџ	Фm№	ХmЁ	Цmё	Чmя	Шmа	Ыmб	Ьmв	Яmг	†mд	£mе	§mжwvзxvиyvйzvк}vл~vмvн	Аvо	Бvп	Вvр	Еvс	Жvт	Йvу	Кvф	Нvх	Оvц	Пvч	Рvш	Уvщ	Фvъ	Хvы	Цvь	Чvэ	Шvю	Ыv€	ЬvА	ЯvБ	†vВ	£vГ	§vД"
	optimizer
 "
trackable_dict_wrapper
-
ґserving_default"
signature_map
"
_tf_keras_input_layer
д
Ј	variables
Єtrainable_variables
єregularization_losses
Ї	keras_api
ї__call__
+Љ&call_and_return_all_conditional_losses

wkernel
xbias
!љ_jit_compiled_convolution_op"
_tf_keras_layer
с
Њ	variables
њtrainable_variables
јregularization_losses
Ѕ	keras_api
¬__call__
+√&call_and_return_all_conditional_losses
	ƒaxis
	ygamma
zbeta
{moving_mean
|moving_variance"
_tf_keras_layer
Ђ
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
д
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses

}kernel
~bias
!—_jit_compiled_convolution_op"
_tf_keras_layer
е
“	variables
”trainable_variables
‘regularization_losses
’	keras_api
÷__call__
+„&call_and_return_all_conditional_losses

kernel
	Аbias
!Ў_jit_compiled_convolution_op"
_tf_keras_layer
х
ў	variables
Џtrainable_variables
џregularization_losses
№	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses
	яaxis

Бgamma
	Вbeta
Гmoving_mean
Дmoving_variance"
_tf_keras_layer
х
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses
	жaxis

Еgamma
	Жbeta
Зmoving_mean
Иmoving_variance"
_tf_keras_layer
Ђ
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
у	variables
фtrainable_variables
хregularization_losses
ц	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"
_tf_keras_layer
х
щ	variables
ъtrainable_variables
ыregularization_losses
ь	keras_api
э__call__
+ю&call_and_return_all_conditional_losses
	€axis

Йgamma
	Кbeta
Лmoving_mean
Мmoving_variance"
_tf_keras_layer
Ђ
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses
Нkernel
	Оbias
!М_jit_compiled_convolution_op"
_tf_keras_layer
х
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
	Уaxis

Пgamma
	Рbeta
Сmoving_mean
Тmoving_variance"
_tf_keras_layer
Ђ
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
Ъ	variables
Ыtrainable_variables
Ьregularization_losses
Э	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses
Уkernel
	Фbias
!†_jit_compiled_convolution_op"
_tf_keras_layer
ж
°	variables
Ґtrainable_variables
£regularization_losses
§	keras_api
•__call__
+¶&call_and_return_all_conditional_losses
Хkernel
	Цbias
!І_jit_compiled_convolution_op"
_tf_keras_layer
х
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses
	Ѓaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance"
_tf_keras_layer
х
ѓ	variables
∞trainable_variables
±regularization_losses
≤	keras_api
≥__call__
+і&call_and_return_all_conditional_losses
	µaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance"
_tf_keras_layer
Ђ
ґ	variables
Јtrainable_variables
Єregularization_losses
є	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
¬	variables
√trainable_variables
ƒregularization_losses
≈	keras_api
∆__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
х
»	variables
…trainable_variables
 regularization_losses
Ћ	keras_api
ћ__call__
+Ќ&call_and_return_all_conditional_losses
	ќaxis

Яgamma
	†beta
°moving_mean
Ґmoving_variance"
_tf_keras_layer
Ђ
ѕ	variables
–trainable_variables
—regularization_losses
“	keras_api
”__call__
+‘&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
’	variables
÷trainable_variables
„regularization_losses
Ў	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses
£kernel
	§bias
!џ_jit_compiled_convolution_op"
_tf_keras_layer
Ђ
w0
x1
y2
z3
{4
|5
}6
~7
8
А9
Б10
В11
Г12
Д13
Е14
Ж15
З16
И17
Й18
К19
Л20
М21
Н22
О23
П24
Р25
С26
Т27
У28
Ф29
Х30
Ц31
Ч32
Ш33
Щ34
Ъ35
Ы36
Ь37
Э38
Ю39
Я40
†41
°42
Ґ43
£44
§45"
trackable_list_wrapper
Э
w0
x1
y2
z3
}4
~5
6
А7
Б8
В9
Е10
Ж11
Й12
К13
Н14
О15
П16
Р17
У18
Ф19
Х20
Ц21
Ч22
Ш23
Ы24
Ь25
Я26
†27
£28
§29"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
÷
бtrace_0
вtrace_1
гtrace_2
дtrace_32г
&__inference_ResNet_layer_call_fn_32179
&__inference_ResNet_layer_call_fn_35523
&__inference_ResNet_layer_call_fn_35620
&__inference_ResNet_layer_call_fn_32723ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zбtrace_0zвtrace_1zгtrace_2zдtrace_3
¬
еtrace_0
жtrace_1
зtrace_2
иtrace_32ѕ
A__inference_ResNet_layer_call_and_return_conditional_losses_35789
A__inference_ResNet_layer_call_and_return_conditional_losses_35958
A__inference_ResNet_layer_call_and_return_conditional_losses_32844
A__inference_ResNet_layer_call_and_return_conditional_losses_32965ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zеtrace_0zжtrace_1zзtrace_2zиtrace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
Р
оtrace_02с
>__inference_random_affine_transform_params_layer_call_fn_35965Ѓ
•≤°
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsҐ
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zоtrace_0
Ђ
пtrace_02М
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_36047Ѓ
•≤°
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsҐ
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zпtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
≤
хtrace_02У
@__inference_image_projective_transform_layer_layer_call_fn_36053ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zхtrace_0
Ќ
цtrace_02Ѓ
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_36061ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zцtrace_0
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
≤
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
і
ьtrace_02Х
B__inference_image_projective_transform_layer_1_layer_call_fn_36067ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zьtrace_0
ѕ
эtrace_02∞
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_36075ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zэtrace_0
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
≤
юnon_trainable_variables
€layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
о
Гtrace_02ѕ
(__inference_add_loss_layer_call_fn_36081Ґ
Щ≤Х
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
annotations™ *
 zГtrace_0
Й
Дtrace_02к
C__inference_add_loss_layer_call_and_return_conditional_losses_36086Ґ
Щ≤Х
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
annotations™ *
 zДtrace_0
*:(А2conv2d_1/kernel
:А2conv2d_1/bias
*:(А2batch_normalization_1/gamma
):'А2batch_normalization_1/beta
2:0А (2!batch_normalization_1/moving_mean
6:4А (2%batch_normalization_1/moving_variance
+:)АА2conv2d_2/kernel
:А2conv2d_2/bias
(:&А2conv2d/kernel
:А2conv2d/bias
*:(А2batch_normalization_2/gamma
):'А2batch_normalization_2/beta
2:0А (2!batch_normalization_2/moving_mean
6:4А (2%batch_normalization_2/moving_variance
(:&А2batch_normalization/gamma
':%А2batch_normalization/beta
0:.А (2batch_normalization/moving_mean
4:2А (2#batch_normalization/moving_variance
*:(А2batch_normalization_3/gamma
):'А2batch_normalization_3/beta
2:0А (2!batch_normalization_3/moving_mean
6:4А (2%batch_normalization_3/moving_variance
*:(А@2conv2d_4/kernel
:@2conv2d_4/bias
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
*:(А@2conv2d_3/kernel
:@2conv2d_3/bias
):'@2batch_normalization_6/gamma
(:&@2batch_normalization_6/beta
1:/@ (2!batch_normalization_6/moving_mean
5:3@ (2%batch_normalization_6/moving_variance
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
):'@2conv2d_6/kernel
:2conv2d_6/bias
§
{0
|1
Г2
Д3
З4
И5
Л6
М7
С8
Т9
Щ10
Ъ11
Э12
Ю13
°14
Ґ15"
trackable_list_wrapper
ю
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
28"
trackable_list_wrapper
(
Е0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
шBх
%__inference_model_layer_call_fn_33370input_1"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
чBф
%__inference_model_layer_call_fn_34492inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
чBф
%__inference_model_layer_call_fn_34592inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
шBх
%__inference_model_layer_call_fn_33895input_1"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ТBП
@__inference_model_layer_call_and_return_conditional_losses_35009inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ТBП
@__inference_model_layer_call_and_return_conditional_losses_35426inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
УBР
@__inference_model_layer_call_and_return_conditional_losses_34091input_1"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
УBР
@__inference_model_layer_call_and_return_conditional_losses_34287input_1"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
 B«
#__inference_signature_wrapper_34392input_1"Ф
Н≤Й
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
annotations™ *
 
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
Ј	variables
Єtrainable_variables
єregularization_losses
ї__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
о
Лtrace_02ѕ
(__inference_conv2d_1_layer_call_fn_36095Ґ
Щ≤Х
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
annotations™ *
 zЛtrace_0
Й
Мtrace_02к
C__inference_conv2d_1_layer_call_and_return_conditional_losses_36105Ґ
Щ≤Х
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
annotations™ *
 zМtrace_0
і2±Ѓ
£≤Я
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
annotations™ *
 0
<
y0
z1
{2
|3"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Њ	variables
њtrainable_variables
јregularization_losses
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
а
Тtrace_0
Уtrace_12•
5__inference_batch_normalization_1_layer_call_fn_36118
5__inference_batch_normalization_1_layer_call_fn_36131і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zТtrace_0zУtrace_1
Ц
Фtrace_0
Хtrace_12џ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36149
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36167і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zФtrace_0zХtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
у
Ыtrace_02‘
-__inference_leaky_re_lu_1_layer_call_fn_36172Ґ
Щ≤Х
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
annotations™ *
 zЫtrace_0
О
Ьtrace_02п
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_36177Ґ
Щ≤Х
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
annotations™ *
 zЬtrace_0
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
Є
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
о
Ґtrace_02ѕ
(__inference_conv2d_2_layer_call_fn_36186Ґ
Щ≤Х
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
annotations™ *
 zҐtrace_0
Й
£trace_02к
C__inference_conv2d_2_layer_call_and_return_conditional_losses_36196Ґ
Щ≤Х
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
annotations™ *
 z£trace_0
і2±Ѓ
£≤Я
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
annotations™ *
 0
/
0
А1"
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
“	variables
”trainable_variables
‘regularization_losses
÷__call__
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
м
©trace_02Ќ
&__inference_conv2d_layer_call_fn_36205Ґ
Щ≤Х
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
annotations™ *
 z©trace_0
З
™trace_02и
A__inference_conv2d_layer_call_and_return_conditional_losses_36215Ґ
Щ≤Х
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
annotations™ *
 z™trace_0
і2±Ѓ
£≤Я
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
annotations™ *
 0
@
Б0
В1
Г2
Д3"
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
ў	variables
Џtrainable_variables
џregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
а
∞trace_0
±trace_12•
5__inference_batch_normalization_2_layer_call_fn_36228
5__inference_batch_normalization_2_layer_call_fn_36241і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z∞trace_0z±trace_1
Ц
≤trace_0
≥trace_12џ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36259
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36277і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z≤trace_0z≥trace_1
 "
trackable_list_wrapper
@
Е0
Ж1
З2
И3"
trackable_list_wrapper
0
Е0
Ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
№
єtrace_0
Їtrace_12°
3__inference_batch_normalization_layer_call_fn_36290
3__inference_batch_normalization_layer_call_fn_36303і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zєtrace_0zЇtrace_1
Т
їtrace_0
Љtrace_12„
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36321
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36339і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zїtrace_0zЉtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
љnon_trainable_variables
Њlayers
њmetrics
 јlayer_regularization_losses
Ѕlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
у
¬trace_02‘
-__inference_leaky_re_lu_2_layer_call_fn_36344Ґ
Щ≤Х
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
annotations™ *
 z¬trace_0
О
√trace_02п
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_36349Ґ
Щ≤Х
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
annotations™ *
 z√trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ƒnon_trainable_variables
≈layers
∆metrics
 «layer_regularization_losses
»layer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
с
…trace_02“
+__inference_leaky_re_lu_layer_call_fn_36354Ґ
Щ≤Х
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
annotations™ *
 z…trace_0
М
 trace_02н
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_36359Ґ
Щ≤Х
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
annotations™ *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ћnon_trainable_variables
ћlayers
Ќmetrics
 ќlayer_regularization_losses
ѕlayer_metrics
у	variables
фtrainable_variables
хregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
й
–trace_02 
#__inference_add_layer_call_fn_36365Ґ
Щ≤Х
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
annotations™ *
 z–trace_0
Д
—trace_02е
>__inference_add_layer_call_and_return_conditional_losses_36371Ґ
Щ≤Х
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
annotations™ *
 z—trace_0
@
Й0
К1
Л2
М3"
trackable_list_wrapper
0
Й0
К1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
щ	variables
ъtrainable_variables
ыregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
а
„trace_0
Ўtrace_12•
5__inference_batch_normalization_3_layer_call_fn_36384
5__inference_batch_normalization_3_layer_call_fn_36397і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z„trace_0zЎtrace_1
Ц
ўtrace_0
Џtrace_12џ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36415
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36433і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zўtrace_0zЏtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
џnon_trainable_variables
№layers
Ёmetrics
 ёlayer_regularization_losses
яlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
у
аtrace_02‘
-__inference_leaky_re_lu_3_layer_call_fn_36438Ґ
Щ≤Х
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
annotations™ *
 zаtrace_0
О
бtrace_02п
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_36443Ґ
Щ≤Х
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
annotations™ *
 zбtrace_0
0
Н0
О1"
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
о
зtrace_02ѕ
(__inference_conv2d_4_layer_call_fn_36452Ґ
Щ≤Х
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
annotations™ *
 zзtrace_0
Й
иtrace_02к
C__inference_conv2d_4_layer_call_and_return_conditional_losses_36462Ґ
Щ≤Х
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
annotations™ *
 zиtrace_0
і2±Ѓ
£≤Я
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
annotations™ *
 0
@
П0
Р1
С2
Т3"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
а
оtrace_0
пtrace_12•
5__inference_batch_normalization_5_layer_call_fn_36475
5__inference_batch_normalization_5_layer_call_fn_36488і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zоtrace_0zпtrace_1
Ц
рtrace_0
сtrace_12џ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36506
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36524і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zрtrace_0zсtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
у
чtrace_02‘
-__inference_leaky_re_lu_5_layer_call_fn_36529Ґ
Щ≤Х
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
annotations™ *
 zчtrace_0
О
шtrace_02п
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_36534Ґ
Щ≤Х
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
annotations™ *
 zшtrace_0
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
Є
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
Ъ	variables
Ыtrainable_variables
Ьregularization_losses
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
о
юtrace_02ѕ
(__inference_conv2d_5_layer_call_fn_36543Ґ
Щ≤Х
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
annotations™ *
 zюtrace_0
Й
€trace_02к
C__inference_conv2d_5_layer_call_and_return_conditional_losses_36553Ґ
Щ≤Х
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
annotations™ *
 z€trace_0
і2±Ѓ
£≤Я
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
annotations™ *
 0
0
Х0
Ц1"
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
°	variables
Ґtrainable_variables
£regularization_losses
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
о
Еtrace_02ѕ
(__inference_conv2d_3_layer_call_fn_36562Ґ
Щ≤Х
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
annotations™ *
 zЕtrace_0
Й
Жtrace_02к
C__inference_conv2d_3_layer_call_and_return_conditional_losses_36572Ґ
Щ≤Х
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
annotations™ *
 zЖtrace_0
і2±Ѓ
£≤Я
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
annotations™ *
 0
@
Ч0
Ш1
Щ2
Ъ3"
trackable_list_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
а
Мtrace_0
Нtrace_12•
5__inference_batch_normalization_6_layer_call_fn_36585
5__inference_batch_normalization_6_layer_call_fn_36598і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zМtrace_0zНtrace_1
Ц
Оtrace_0
Пtrace_12џ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36616
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36634і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zОtrace_0zПtrace_1
 "
trackable_list_wrapper
@
Ы0
Ь1
Э2
Ю3"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
ѓ	variables
∞trainable_variables
±regularization_losses
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
а
Хtrace_0
Цtrace_12•
5__inference_batch_normalization_4_layer_call_fn_36647
5__inference_batch_normalization_4_layer_call_fn_36660і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zХtrace_0zЦtrace_1
Ц
Чtrace_0
Шtrace_12џ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36678
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36696і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zЧtrace_0zШtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
ґ	variables
Јtrainable_variables
Єregularization_losses
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
у
Юtrace_02‘
-__inference_leaky_re_lu_6_layer_call_fn_36701Ґ
Щ≤Х
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
annotations™ *
 zЮtrace_0
О
Яtrace_02п
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_36706Ґ
Щ≤Х
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
annotations™ *
 zЯtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
у
•trace_02‘
-__inference_leaky_re_lu_4_layer_call_fn_36711Ґ
Щ≤Х
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
annotations™ *
 z•trace_0
О
¶trace_02п
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_36716Ґ
Щ≤Х
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
annotations™ *
 z¶trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Іnon_trainable_variables
®layers
©metrics
 ™layer_regularization_losses
Ђlayer_metrics
¬	variables
√trainable_variables
ƒregularization_losses
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
л
ђtrace_02ћ
%__inference_add_1_layer_call_fn_36722Ґ
Щ≤Х
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
annotations™ *
 zђtrace_0
Ж
≠trace_02з
@__inference_add_1_layer_call_and_return_conditional_losses_36728Ґ
Щ≤Х
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
annotations™ *
 z≠trace_0
@
Я0
†1
°2
Ґ3"
trackable_list_wrapper
0
Я0
†1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ѓnon_trainable_variables
ѓlayers
∞metrics
 ±layer_regularization_losses
≤layer_metrics
»	variables
…trainable_variables
 regularization_losses
ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
а
≥trace_0
іtrace_12•
5__inference_batch_normalization_7_layer_call_fn_36741
5__inference_batch_normalization_7_layer_call_fn_36754і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 z≥trace_0zіtrace_1
Ц
µtrace_0
ґtrace_12џ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36772
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36790і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 zµtrace_0zґtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
ѕ	variables
–trainable_variables
—regularization_losses
”__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
у
Љtrace_02‘
-__inference_leaky_re_lu_7_layer_call_fn_36795Ґ
Щ≤Х
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
annotations™ *
 zЉtrace_0
О
љtrace_02п
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_36800Ґ
Щ≤Х
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
annotations™ *
 zљtrace_0
0
£0
§1"
trackable_list_wrapper
0
£0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
’	variables
÷trainable_variables
„regularization_losses
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
о
√trace_02ѕ
(__inference_conv2d_6_layer_call_fn_36809Ґ
Щ≤Х
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
annotations™ *
 z√trace_0
Й
ƒtrace_02к
C__inference_conv2d_6_layer_call_and_return_conditional_losses_36820Ґ
Щ≤Х
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
annotations™ *
 zƒtrace_0
і2±Ѓ
£≤Я
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
annotations™ *
 0
§
{0
|1
Г2
Д3
З4
И5
Л6
М7
С8
Т9
Щ10
Ъ11
Э12
Ю13
°14
Ґ15"
trackable_list_wrapper
ж
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513
614
715
816
917
:18
;19
<20
=21
>22
?23
@24
A25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
&__inference_ResNet_layer_call_fn_32179input_2"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
шBх
&__inference_ResNet_layer_call_fn_35523inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
шBх
&__inference_ResNet_layer_call_fn_35620inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
щBц
&__inference_ResNet_layer_call_fn_32723input_2"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
УBР
A__inference_ResNet_layer_call_and_return_conditional_losses_35789inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
УBР
A__inference_ResNet_layer_call_and_return_conditional_losses_35958inputs"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ФBС
A__inference_ResNet_layer_call_and_return_conditional_losses_32844input_2"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ФBС
A__inference_ResNet_layer_call_and_return_conditional_losses_32965input_2"ј
Ј≤≥
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
ыBш
>__inference_random_affine_transform_params_layer_call_fn_35965inp"Ѓ
•≤°
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsҐ
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЦBУ
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_36047inp"Ѓ
•≤°
FullArgSpec#
argsЪ
jself
jinp
jWIDTH
varargs
 
varkw
 
defaultsҐ
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ђB©
@__inference_image_projective_transform_layer_layer_call_fn_36053inputs
transforms"ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
«Bƒ
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_36061inputs
transforms"ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЃBЂ
B__inference_image_projective_transform_layer_1_layer_call_fn_36067inputs
transforms"ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
…B∆
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_36075inputs
transforms"ќ
≈≤Ѕ
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
defaultsҐ

`А
`А

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
№Bў
(__inference_add_loss_layer_call_fn_36081inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_add_loss_layer_call_and_return_conditional_losses_36086inputs"Ґ
Щ≤Х
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
annotations™ *
 
R
≈	variables
∆	keras_api

«total

»count"
_tf_keras_metric
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
№Bў
(__inference_conv2d_1_layer_call_fn_36095inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_conv2d_1_layer_call_and_return_conditional_losses_36105inputs"Ґ
Щ≤Х
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
annotations™ *
 
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
5__inference_batch_normalization_1_layer_call_fn_36118inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ыBш
5__inference_batch_normalization_1_layer_call_fn_36131inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36149inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36167inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
бBё
-__inference_leaky_re_lu_1_layer_call_fn_36172inputs"Ґ
Щ≤Х
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
annotations™ *
 
ьBщ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_36177inputs"Ґ
Щ≤Х
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
annotations™ *
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
№Bў
(__inference_conv2d_2_layer_call_fn_36186inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_conv2d_2_layer_call_and_return_conditional_losses_36196inputs"Ґ
Щ≤Х
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
annotations™ *
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
ЏB„
&__inference_conv2d_layer_call_fn_36205inputs"Ґ
Щ≤Х
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
annotations™ *
 
хBт
A__inference_conv2d_layer_call_and_return_conditional_losses_36215inputs"Ґ
Щ≤Х
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
annotations™ *
 
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
5__inference_batch_normalization_2_layer_call_fn_36228inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ыBш
5__inference_batch_normalization_2_layer_call_fn_36241inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36259inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36277inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
0
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
3__inference_batch_normalization_layer_call_fn_36290inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
щBц
3__inference_batch_normalization_layer_call_fn_36303inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ФBС
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36321inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ФBС
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36339inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
бBё
-__inference_leaky_re_lu_2_layer_call_fn_36344inputs"Ґ
Щ≤Х
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
annotations™ *
 
ьBщ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_36349inputs"Ґ
Щ≤Х
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
annotations™ *
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
яB№
+__inference_leaky_re_lu_layer_call_fn_36354inputs"Ґ
Щ≤Х
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
annotations™ *
 
ъBч
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_36359inputs"Ґ
Щ≤Х
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
annotations™ *
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
гBа
#__inference_add_layer_call_fn_36365inputs/0inputs/1"Ґ
Щ≤Х
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
annotations™ *
 
юBы
>__inference_add_layer_call_and_return_conditional_losses_36371inputs/0inputs/1"Ґ
Щ≤Х
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
annotations™ *
 
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
5__inference_batch_normalization_3_layer_call_fn_36384inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ыBш
5__inference_batch_normalization_3_layer_call_fn_36397inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36415inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36433inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
бBё
-__inference_leaky_re_lu_3_layer_call_fn_36438inputs"Ґ
Щ≤Х
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
annotations™ *
 
ьBщ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_36443inputs"Ґ
Щ≤Х
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
annotations™ *
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
№Bў
(__inference_conv2d_4_layer_call_fn_36452inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_conv2d_4_layer_call_and_return_conditional_losses_36462inputs"Ґ
Щ≤Х
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
annotations™ *
 
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
5__inference_batch_normalization_5_layer_call_fn_36475inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ыBш
5__inference_batch_normalization_5_layer_call_fn_36488inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36506inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36524inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
бBё
-__inference_leaky_re_lu_5_layer_call_fn_36529inputs"Ґ
Щ≤Х
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
annotations™ *
 
ьBщ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_36534inputs"Ґ
Щ≤Х
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
annotations™ *
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
№Bў
(__inference_conv2d_5_layer_call_fn_36543inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_conv2d_5_layer_call_and_return_conditional_losses_36553inputs"Ґ
Щ≤Х
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
annotations™ *
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
№Bў
(__inference_conv2d_3_layer_call_fn_36562inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_conv2d_3_layer_call_and_return_conditional_losses_36572inputs"Ґ
Щ≤Х
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
annotations™ *
 
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
5__inference_batch_normalization_6_layer_call_fn_36585inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ыBш
5__inference_batch_normalization_6_layer_call_fn_36598inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36616inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36634inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
5__inference_batch_normalization_4_layer_call_fn_36647inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ыBш
5__inference_batch_normalization_4_layer_call_fn_36660inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36678inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36696inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
бBё
-__inference_leaky_re_lu_6_layer_call_fn_36701inputs"Ґ
Щ≤Х
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
annotations™ *
 
ьBщ
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_36706inputs"Ґ
Щ≤Х
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
annotations™ *
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
бBё
-__inference_leaky_re_lu_4_layer_call_fn_36711inputs"Ґ
Щ≤Х
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
annotations™ *
 
ьBщ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_36716inputs"Ґ
Щ≤Х
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
annotations™ *
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
еBв
%__inference_add_1_layer_call_fn_36722inputs/0inputs/1"Ґ
Щ≤Х
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
annotations™ *
 
АBэ
@__inference_add_1_layer_call_and_return_conditional_losses_36728inputs/0inputs/1"Ґ
Щ≤Х
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
annotations™ *
 
0
°0
Ґ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
5__inference_batch_normalization_7_layer_call_fn_36741inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ыBш
5__inference_batch_normalization_7_layer_call_fn_36754inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36772inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЦBУ
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36790inputs"і
Ђ≤І
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

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
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
бBё
-__inference_leaky_re_lu_7_layer_call_fn_36795inputs"Ґ
Щ≤Х
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
annotations™ *
 
ьBщ
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_36800inputs"Ґ
Щ≤Х
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
annotations™ *
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
№Bў
(__inference_conv2d_6_layer_call_fn_36809inputs"Ґ
Щ≤Х
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
annotations™ *
 
чBф
C__inference_conv2d_6_layer_call_and_return_conditional_losses_36820inputs"Ґ
Щ≤Х
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
annotations™ *
 
0
«0
»1"
trackable_list_wrapper
.
≈	variables"
_generic_user_object
:  (2total
:  (2count
/:-А2Adam/conv2d_1/kernel/m
!:А2Adam/conv2d_1/bias/m
/:-А2"Adam/batch_normalization_1/gamma/m
.:,А2!Adam/batch_normalization_1/beta/m
0:.АА2Adam/conv2d_2/kernel/m
!:А2Adam/conv2d_2/bias/m
-:+А2Adam/conv2d/kernel/m
:А2Adam/conv2d/bias/m
/:-А2"Adam/batch_normalization_2/gamma/m
.:,А2!Adam/batch_normalization_2/beta/m
-:+А2 Adam/batch_normalization/gamma/m
,:*А2Adam/batch_normalization/beta/m
/:-А2"Adam/batch_normalization_3/gamma/m
.:,А2!Adam/batch_normalization_3/beta/m
/:-А@2Adam/conv2d_4/kernel/m
 :@2Adam/conv2d_4/bias/m
.:,@2"Adam/batch_normalization_5/gamma/m
-:+@2!Adam/batch_normalization_5/beta/m
.:,@@2Adam/conv2d_5/kernel/m
 :@2Adam/conv2d_5/bias/m
/:-А@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
.:,@2"Adam/batch_normalization_6/gamma/m
-:+@2!Adam/batch_normalization_6/beta/m
.:,@2"Adam/batch_normalization_4/gamma/m
-:+@2!Adam/batch_normalization_4/beta/m
.:,@2"Adam/batch_normalization_7/gamma/m
-:+@2!Adam/batch_normalization_7/beta/m
.:,@2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
/:-А2Adam/conv2d_1/kernel/v
!:А2Adam/conv2d_1/bias/v
/:-А2"Adam/batch_normalization_1/gamma/v
.:,А2!Adam/batch_normalization_1/beta/v
0:.АА2Adam/conv2d_2/kernel/v
!:А2Adam/conv2d_2/bias/v
-:+А2Adam/conv2d/kernel/v
:А2Adam/conv2d/bias/v
/:-А2"Adam/batch_normalization_2/gamma/v
.:,А2!Adam/batch_normalization_2/beta/v
-:+А2 Adam/batch_normalization/gamma/v
,:*А2Adam/batch_normalization/beta/v
/:-А2"Adam/batch_normalization_3/gamma/v
.:,А2!Adam/batch_normalization_3/beta/v
/:-А@2Adam/conv2d_4/kernel/v
 :@2Adam/conv2d_4/bias/v
.:,@2"Adam/batch_normalization_5/gamma/v
-:+@2!Adam/batch_normalization_5/beta/v
.:,@@2Adam/conv2d_5/kernel/v
 :@2Adam/conv2d_5/bias/v
/:-А@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
.:,@2"Adam/batch_normalization_6/gamma/v
-:+@2!Adam/batch_normalization_6/beta/v
.:,@2"Adam/batch_normalization_4/gamma/v
-:+@2!Adam/batch_normalization_4/beta/v
.:,@2"Adam/batch_normalization_7/gamma/v
-:+@2!Adam/batch_normalization_7/beta/v
.:,@2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
J
Constjtf.TrackableConstant∞
A__inference_ResNet_layer_call_and_return_conditional_losses_32844кSwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§RҐO
HҐE
;К8
input_2+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∞
A__inference_ResNet_layer_call_and_return_conditional_losses_32965кSwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§RҐO
HҐE
;К8
input_2+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
A__inference_ResNet_layer_call_and_return_conditional_losses_35789йSwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
A__inference_ResNet_layer_call_and_return_conditional_losses_35958йSwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ И
&__inference_ResNet_layer_call_fn_32179ЁSwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§RҐO
HҐE
;К8
input_2+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€И
&__inference_ResNet_layer_call_fn_32723ЁSwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§RҐO
HҐE
;К8
input_2+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€З
&__inference_ResNet_layer_call_fn_35523№Swxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€З
&__inference_ResNet_layer_call_fn_35620№Swxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§QҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€У
 __inference__wrapped_model_31307оUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕJҐG
@Ґ=
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "I™F
D
ResNet:К7
ResNet+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ъ
@__inference_add_1_layer_call_and_return_conditional_losses_36728’СҐН
ЕҐБ
Ъ|
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
<К9
inputs/1+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ т
%__inference_add_1_layer_call_fn_36722»СҐН
ЕҐБ
Ъ|
<К9
inputs/0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
<К9
inputs/1+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ь
>__inference_add_layer_call_and_return_conditional_losses_36371ўФҐР
ИҐД
БЪ~
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
=К:
inputs/1,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ф
#__inference_add_layer_call_fn_36365ћФҐР
ИҐД
БЪ~
=К:
inputs/0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
=К:
inputs/1,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЛ
C__inference_add_loss_layer_call_and_return_conditional_losses_36086DҐ
Ґ
К
inputs 
™ ""Ґ

К
0 
Ъ
К	
1/0 U
(__inference_add_loss_layer_call_fn_36081)Ґ
Ґ
К
inputs 
™ "К н
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36149Шyz{|NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ н
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_36167Шyz{|NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≈
5__inference_batch_normalization_1_layer_call_fn_36118Лyz{|NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А≈
5__inference_batch_normalization_1_layer_call_fn_36131Лyz{|NҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ас
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36259ЬБВГДNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ с
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_36277ЬБВГДNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ …
5__inference_batch_normalization_2_layer_call_fn_36228ПБВГДNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А…
5__inference_batch_normalization_2_layer_call_fn_36241ПБВГДNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ас
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36415ЬЙКЛМNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ с
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_36433ЬЙКЛМNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ …
5__inference_batch_normalization_3_layer_call_fn_36384ПЙКЛМNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А…
5__inference_batch_normalization_3_layer_call_fn_36397ПЙКЛМNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ап
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36678ЪЫЬЭЮMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ п
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36696ЪЫЬЭЮMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ «
5__inference_batch_normalization_4_layer_call_fn_36647НЫЬЭЮMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@«
5__inference_batch_normalization_4_layer_call_fn_36660НЫЬЭЮMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@п
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36506ЪПРСТMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ п
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_36524ЪПРСТMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ «
5__inference_batch_normalization_5_layer_call_fn_36475НПРСТMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@«
5__inference_batch_normalization_5_layer_call_fn_36488НПРСТMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@п
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36616ЪЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ п
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_36634ЪЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ «
5__inference_batch_normalization_6_layer_call_fn_36585НЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@«
5__inference_batch_normalization_6_layer_call_fn_36598НЧШЩЪMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@п
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36772ЪЯ†°ҐMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ п
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_36790ЪЯ†°ҐMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ «
5__inference_batch_normalization_7_layer_call_fn_36741НЯ†°ҐMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@«
5__inference_batch_normalization_7_layer_call_fn_36754НЯ†°ҐMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@п
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36321ЬЕЖЗИNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ п
N__inference_batch_normalization_layer_call_and_return_conditional_losses_36339ЬЕЖЗИNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ «
3__inference_batch_normalization_layer_call_fn_36290ПЕЖЗИNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А«
3__inference_batch_normalization_layer_call_fn_36303ПЕЖЗИNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аў
C__inference_conv2d_1_layer_call_and_return_conditional_losses_36105СwxIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ±
(__inference_conv2d_1_layer_call_fn_36095ДwxIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЏ
C__inference_conv2d_2_layer_call_and_return_conditional_losses_36196Т}~JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≤
(__inference_conv2d_2_layer_call_fn_36186Е}~JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аџ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_36572УХЦJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≥
(__inference_conv2d_3_layer_call_fn_36562ЖХЦJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@џ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_36462УНОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≥
(__inference_conv2d_4_layer_call_fn_36452ЖНОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_36553ТУФIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≤
(__inference_conv2d_5_layer_call_fn_36543ЕУФIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
C__inference_conv2d_6_layer_call_and_return_conditional_losses_36820Т£§IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≤
(__inference_conv2d_6_layer_call_fn_36809Е£§IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Ў
A__inference_conv2d_layer_call_and_return_conditional_losses_36215ТАIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ∞
&__inference_conv2d_layer_call_fn_36205ЕАIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аю
]__inference_image_projective_transform_layer_1_layer_call_and_return_conditional_losses_36075ЬiҐf
_Ґ\
*К'
inputs€€€€€€€€€цц
$К!

transforms€€€€€€€€€
`А
`А
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ ÷
B__inference_image_projective_transform_layer_1_layer_call_fn_36067ПiҐf
_Ґ\
*К'
inputs€€€€€€€€€цц
$К!

transforms€€€€€€€€€
`А
`А
™ ""К€€€€€€€€€ААМ
[__inference_image_projective_transform_layer_layer_call_and_return_conditional_losses_36061ђyҐv
oҐl
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
$К!

transforms€€€€€€€€€
`А
`А
™ "/Ґ,
%К"
0€€€€€€€€€цц
Ъ д
@__inference_image_projective_transform_layer_layer_call_fn_36053ЯyҐv
oҐl
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
$К!

transforms€€€€€€€€€
`А
`А
™ ""К€€€€€€€€€ццџ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_36177ОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≥
-__inference_leaky_re_lu_1_layer_call_fn_36172БJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аџ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_36349ОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≥
-__inference_leaky_re_lu_2_layer_call_fn_36344БJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аџ
H__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_36443ОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≥
-__inference_leaky_re_lu_3_layer_call_fn_36438БJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аў
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_36716МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
-__inference_leaky_re_lu_4_layer_call_fn_36711IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ў
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_36534МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
-__inference_leaky_re_lu_5_layer_call_fn_36529IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ў
H__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_36706МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
-__inference_leaky_re_lu_6_layer_call_fn_36701IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ў
H__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_36800МIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
-__inference_leaky_re_lu_7_layer_call_fn_36795IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ў
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_36359ОJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ±
+__inference_leaky_re_lu_layer_call_fn_36354БJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ањ
@__inference_model_layer_call_and_return_conditional_losses_34091ъUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕRҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "MҐJ
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ
К	
1/0 њ
@__inference_model_layer_call_and_return_conditional_losses_34287ъUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕRҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "MҐJ
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ
К	
1/0 Њ
@__inference_model_layer_call_and_return_conditional_losses_35009щUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "MҐJ
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ
К	
1/0 Њ
@__inference_model_layer_call_and_return_conditional_losses_35426щUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "MҐJ
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ
К	
1/0 Й
%__inference_model_layer_call_fn_33370яUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕRҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Й
%__inference_model_layer_call_fn_33895яUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕRҐO
HҐE
;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€И
%__inference_model_layer_call_fn_34492ёUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€И
%__inference_model_layer_call_fn_34592ёUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕQҐN
GҐD
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ш
Y__inference_random_affine_transform_params_layer_call_and_return_conditional_losses_36047ЪKҐH
AҐ>
7К4
inp+€€€€€€€€€€€€€€€€€€€€€€€€€€€
`А
™ "KҐH
AҐ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ ѕ
>__inference_random_affine_transform_params_layer_call_fn_35965МKҐH
AҐ>
7К4
inp+€€€€€€€€€€€€€€€€€€€€€€€€€€€
`А
™ "=Ґ:
К
0€€€€€€€€€
К
1€€€€€€€€€°
#__inference_signature_wrapper_34392щUwxyz{|А}~ЕЖЗИБВГДЙКЛМНОПРСТХЦУФЫЬЭЮЧШЩЪЯ†°Ґ£§ЕUҐR
Ґ 
K™H
F
input_1;К8
input_1+€€€€€€€€€€€€€€€€€€€€€€€€€€€"I™F
D
ResNet:К7
ResNet+€€€€€€€€€€€€€€€€€€€€€€€€€€€