ри
ђЩ
9
Add
x"T
y"T
z"T"
Ttype:
2	
б
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
A
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
:
Greater
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.1.02v1.1.0-rc0-61-g1ec6ed5лЈ
]
PlaceholderPlaceholder*'
_output_shapes
:џџџџџџџџџ	*
shape: *
dtype0
_
Placeholder_1Placeholder*
shape: *
dtype0*'
_output_shapes
:џџџџџџџџџ

-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"	       *
_output_shapes
:*
dtype0

+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *нУО*
_output_shapes
: *
dtype0

+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *нУ>*
dtype0*
_output_shapes
: 
х
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@dense/kernel*
seed2*
dtype0*
_output_shapes

:	 
Ю
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
р
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:	 
в
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:	 
Ё
dense/kernel
VariableV2*
	container *
dtype0*
_class
loc:@dense/kernel*
_output_shapes

:	 *
shape
:	 *
shared_name 
Ч
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:	 
u
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes

:	 

dense/bias/Initializer/ConstConst*
_class
loc:@dense/bias*
valueB *    *
dtype0*
_output_shapes
: 


dense/bias
VariableV2*
_class
loc:@dense/bias*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
В
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
: 

dense/MatMulMatMulPlaceholderdense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ *
transpose_a( 

dense/BiasAddBiasAdddense/MatMuldense/bias/read*'
_output_shapes
:џџџџџџџџџ *
T0*
data_formatNHWC
S

dense/ReluReludense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ 
Ѓ
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
valueB"        *
dtype0*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB
 *qФО*
_output_shapes
: *
dtype0

-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
valueB
 *qФ>*
dtype0*
_output_shapes
: 
ы
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0*
dtype0*
seed2*

seed 
ж
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
ш
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:  
к
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0
Ѕ
dense_1/kernel
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape
:  *
dtype0*
_output_shapes

:  
Я
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0*
validate_shape(*
use_locking(
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0

dense_1/bias/Initializer/ConstConst*
_class
loc:@dense_1/bias*
valueB *    *
dtype0*
_output_shapes
: 

dense_1/bias
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
К
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
: 

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ *
transpose_a( 

dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*'
_output_shapes
:џџџџџџџџџ *
T0*
data_formatNHWC
W
dense_2/ReluReludense_2/BiasAdd*'
_output_shapes
:џџџџџџџџџ *
T0
Ѓ
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
valueB"       *
dtype0*
_output_shapes
:

-dense_2/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_2/kernel*
valueB
 *їќгО*
dtype0*
_output_shapes
: 

-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
valueB
 *їќг>*
_output_shapes
: *
dtype0
ы
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
seed2'*
T0*

seed *
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 
ж
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
ш
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0
к
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0
Ѕ
dense_2/kernel
VariableV2*
shared_name *!
_class
loc:@dense_2/kernel*
	container *
shape
: *
dtype0*
_output_shapes

: 
Я
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
{
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0

dense_2/bias/Initializer/ConstConst*
_class
loc:@dense_2/bias*
valueB*    *
_output_shapes
:*
dtype0

dense_2/bias
VariableV2*
	container *
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
:*
shape:*
shared_name 
К
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/Const*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:

dense_3/MatMulMatMuldense_2/Reludense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 

dense_3/BiasAddBiasAdddense_3/MatMuldense_2/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
U
SoftmaxSoftmaxdense_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
m
 softmax_cross_entropy_loss/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
x
.softmax_cross_entropy_loss/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
z
0softmax_cross_entropy_loss/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0softmax_cross_entropy_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

(softmax_cross_entropy_loss/strided_sliceStridedSlice softmax_cross_entropy_loss/Shape.softmax_cross_entropy_loss/strided_slice/stack0softmax_cross_entropy_loss/strided_slice/stack_10softmax_cross_entropy_loss/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0*
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask 

!softmax_cross_entropy_loss/Cast_1Cast(softmax_cross_entropy_loss/strided_slice*
_output_shapes
: *

DstT0*

SrcT0
i
$softmax_cross_entropy_loss/truediv/xConst*
valueB
 *ЌХ'7*
_output_shapes
: *
dtype0

"softmax_cross_entropy_loss/truedivRealDiv$softmax_cross_entropy_loss/truediv/x!softmax_cross_entropy_loss/Cast_1*
_output_shapes
: *
T0
e
 softmax_cross_entropy_loss/mul/yConst*
valueB
 *Xџ?*
dtype0*
_output_shapes
: 

softmax_cross_entropy_loss/mulMulPlaceholder_1 softmax_cross_entropy_loss/mul/y*
T0*'
_output_shapes
:џџџџџџџџџ

softmax_cross_entropy_loss/addAddsoftmax_cross_entropy_loss/mul"softmax_cross_entropy_loss/truediv*'
_output_shapes
:џџџџџџџџџ*
T0
a
softmax_cross_entropy_loss/RankConst*
value	B :*
dtype0*
_output_shapes
: 
q
"softmax_cross_entropy_loss/Shape_1Shapedense_3/BiasAdd*
T0*
out_type0*
_output_shapes
:
c
!softmax_cross_entropy_loss/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
q
"softmax_cross_entropy_loss/Shape_2Shapedense_3/BiasAdd*
out_type0*
_output_shapes
:*
T0
b
 softmax_cross_entropy_loss/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0

softmax_cross_entropy_loss/SubSub!softmax_cross_entropy_loss/Rank_1 softmax_cross_entropy_loss/Sub/y*
_output_shapes
: *
T0

&softmax_cross_entropy_loss/Slice/beginPacksoftmax_cross_entropy_loss/Sub*

axis *
_output_shapes
:*
T0*
N
o
%softmax_cross_entropy_loss/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ю
 softmax_cross_entropy_loss/SliceSlice"softmax_cross_entropy_loss/Shape_2&softmax_cross_entropy_loss/Slice/begin%softmax_cross_entropy_loss/Slice/size*
Index0*
T0*
_output_shapes
:
}
*softmax_cross_entropy_loss/concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
h
&softmax_cross_entropy_loss/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
н
!softmax_cross_entropy_loss/concatConcatV2*softmax_cross_entropy_loss/concat/values_0 softmax_cross_entropy_loss/Slice&softmax_cross_entropy_loss/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Њ
"softmax_cross_entropy_loss/ReshapeReshapedense_3/BiasAdd!softmax_cross_entropy_loss/concat*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
c
!softmax_cross_entropy_loss/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 

"softmax_cross_entropy_loss/Shape_3Shapesoftmax_cross_entropy_loss/add*
out_type0*
_output_shapes
:*
T0
d
"softmax_cross_entropy_loss/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_1Sub!softmax_cross_entropy_loss/Rank_2"softmax_cross_entropy_loss/Sub_1/y*
T0*
_output_shapes
: 

(softmax_cross_entropy_loss/Slice_1/beginPack softmax_cross_entropy_loss/Sub_1*

axis *
_output_shapes
:*
T0*
N
q
'softmax_cross_entropy_loss/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
д
"softmax_cross_entropy_loss/Slice_1Slice"softmax_cross_entropy_loss/Shape_3(softmax_cross_entropy_loss/Slice_1/begin'softmax_cross_entropy_loss/Slice_1/size*
_output_shapes
:*
Index0*
T0

,softmax_cross_entropy_loss/concat_1/values_0Const*
valueB:
џџџџџџџџџ*
_output_shapes
:*
dtype0
j
(softmax_cross_entropy_loss/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
х
#softmax_cross_entropy_loss/concat_1ConcatV2,softmax_cross_entropy_loss/concat_1/values_0"softmax_cross_entropy_loss/Slice_1(softmax_cross_entropy_loss/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
Н
$softmax_cross_entropy_loss/Reshape_1Reshapesoftmax_cross_entropy_loss/add#softmax_cross_entropy_loss/concat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
и
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits"softmax_cross_entropy_loss/Reshape$softmax_cross_entropy_loss/Reshape_1*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
d
"softmax_cross_entropy_loss/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/Sub_2Subsoftmax_cross_entropy_loss/Rank"softmax_cross_entropy_loss/Sub_2/y*
T0*
_output_shapes
: 
r
(softmax_cross_entropy_loss/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:

'softmax_cross_entropy_loss/Slice_2/sizePack softmax_cross_entropy_loss/Sub_2*

axis *
_output_shapes
:*
T0*
N
н
"softmax_cross_entropy_loss/Slice_2Slice"softmax_cross_entropy_loss/Shape_1(softmax_cross_entropy_loss/Slice_2/begin'softmax_cross_entropy_loss/Slice_2/size*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0
Д
$softmax_cross_entropy_loss/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy"softmax_cross_entropy_loss/Slice_2*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
 
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
out_type0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
_output_shapes
: *
dtype0
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
Й
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ё
softmax_cross_entropy_loss/MulMul$softmax_cross_entropy_loss/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*#
_output_shapes
:џџџџџџџџџ*
T0
И
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
Ѕ
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
С
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 

1softmax_cross_entropy_loss/num_present/zeros_like	ZerosLike&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes
: *
T0
Ч
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
Щ
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Щ
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
T0*
_output_shapes
: 
ы
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
ь
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
ъ
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 

Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
щ
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
_output_shapes
: *
dtype0
П
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
ц
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeShape$softmax_cross_entropy_loss/Reshape_2L^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
_output_shapes
:*
T0
Ч
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:џџџџџџџџџ
р
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:џџџџџџџџџ
Ф
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
_output_shapes
:*
dtype0
г
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Г
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Љ
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
З
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
_output_shapes
: *
dtype0

"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
Е
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
Л
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
_output_shapes
: *
dtype0
Н
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
_output_shapes
: *
dtype0
Ѕ
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
_output_shapes
: *
T0
М
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 

softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
u
%softmax_cross_entropy_loss/zeros_like	ZerosLike softmax_cross_entropy_loss/Sum_1*
_output_shapes
: *
T0
Ж
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
e
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
T
ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
o
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
P
CastCastEqual*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
O
ConstConst*
valueB: *
_output_shapes
:*
dtype0
W
MeanMeanCastConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 

:gradients/softmax_cross_entropy_loss/value_grad/zeros_like	ZerosLikesoftmax_cross_entropy_loss/div*
T0*
_output_shapes
: 
б
6gradients/softmax_cross_entropy_loss/value_grad/SelectSelect"softmax_cross_entropy_loss/Greatergradients/Fill:gradients/softmax_cross_entropy_loss/value_grad/zeros_like*
_output_shapes
: *
T0
г
8gradients/softmax_cross_entropy_loss/value_grad/Select_1Select"softmax_cross_entropy_loss/Greater:gradients/softmax_cross_entropy_loss/value_grad/zeros_likegradients/Fill*
_output_shapes
: *
T0
М
@gradients/softmax_cross_entropy_loss/value_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_loss/value_grad/Select9^gradients/softmax_cross_entropy_loss/value_grad/Select_1
Л
Hgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_loss/value_grad/SelectA^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_loss/value_grad/Select*
_output_shapes
: 
С
Jgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_loss/value_grad/Select_1A^gradients/softmax_cross_entropy_loss/value_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_loss/value_grad/Select_1*
_output_shapes
: 
v
3gradients/softmax_cross_entropy_loss/div_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
x
5gradients/softmax_cross_entropy_loss/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Cgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/div_grad/Shape5gradients/softmax_cross_entropy_loss/div_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ю
5gradients/softmax_cross_entropy_loss/div_grad/RealDivRealDivHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
є
1gradients/softmax_cross_entropy_loss/div_grad/SumSum5gradients/softmax_cross_entropy_loss/div_grad/RealDivCgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
з
5gradients/softmax_cross_entropy_loss/div_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/div_grad/Sum3gradients/softmax_cross_entropy_loss/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
{
1gradients/softmax_cross_entropy_loss/div_grad/NegNeg softmax_cross_entropy_loss/Sum_1*
T0*
_output_shapes
: 
Й
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1RealDiv1gradients/softmax_cross_entropy_loss/div_grad/Neg!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
П
7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2RealDiv7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
м
1gradients/softmax_cross_entropy_loss/div_grad/mulMulHgradients/softmax_cross_entropy_loss/value_grad/tuple/control_dependency7gradients/softmax_cross_entropy_loss/div_grad/RealDiv_2*
_output_shapes
: *
T0
є
3gradients/softmax_cross_entropy_loss/div_grad/Sum_1Sum1gradients/softmax_cross_entropy_loss/div_grad/mulEgradients/softmax_cross_entropy_loss/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
н
7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/div_grad/Sum_15gradients/softmax_cross_entropy_loss/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
И
>gradients/softmax_cross_entropy_loss/div_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/div_grad/Reshape8^gradients/softmax_cross_entropy_loss/div_grad/Reshape_1
Е
Fgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/div_grad/Reshape?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape*
_output_shapes
: *
T0
Л
Hgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/div_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/div_grad/tuple/group_deps*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/div_grad/Reshape_1*
_output_shapes
: *
T0

;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like	ZerosLike$softmax_cross_entropy_loss/ones_like*
T0*
_output_shapes
: 

7gradients/softmax_cross_entropy_loss/Select_grad/SelectSelect softmax_cross_entropy_loss/EqualHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1;gradients/softmax_cross_entropy_loss/Select_grad/zeros_like*
T0*
_output_shapes
: 

9gradients/softmax_cross_entropy_loss/Select_grad/Select_1Select softmax_cross_entropy_loss/Equal;gradients/softmax_cross_entropy_loss/Select_grad/zeros_likeHgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
П
Agradients/softmax_cross_entropy_loss/Select_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_loss/Select_grad/Select:^gradients/softmax_cross_entropy_loss/Select_grad/Select_1
П
Igradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_loss/Select_grad/SelectB^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select*
_output_shapes
: 
Х
Kgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_loss/Select_grad/Select_1B^gradients/softmax_cross_entropy_loss/Select_grad/tuple/group_deps*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Select_grad/Select_1*
_output_shapes
: *
T0

=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
_output_shapes
: *
dtype0
ј
7gradients/softmax_cross_entropy_loss/Sum_1_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/div_grad/tuple/control_dependency=gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 

>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiplesConst*
valueB *
_output_shapes
: *
dtype0
ш
4gradients/softmax_cross_entropy_loss/Sum_1_grad/TileTile7gradients/softmax_cross_entropy_loss/Sum_1_grad/Reshape>gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile/multiples*

Tmultiples0*
T0*
_output_shapes
: 

;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ц
5gradients/softmax_cross_entropy_loss/Sum_grad/ReshapeReshape4gradients/softmax_cross_entropy_loss/Sum_1_grad/Tile;gradients/softmax_cross_entropy_loss/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

3gradients/softmax_cross_entropy_loss/Sum_grad/ShapeShapesoftmax_cross_entropy_loss/Mul*
T0*
out_type0*
_output_shapes
:
ц
2gradients/softmax_cross_entropy_loss/Sum_grad/TileTile5gradients/softmax_cross_entropy_loss/Sum_grad/Reshape3gradients/softmax_cross_entropy_loss/Sum_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*

Tmultiples0

Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0

=gradients/softmax_cross_entropy_loss/num_present_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/Select_grad/tuple/control_dependency_1Cgradients/softmax_cross_entropy_loss/num_present_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
Г
;gradients/softmax_cross_entropy_loss/num_present_grad/ShapeShape8softmax_cross_entropy_loss/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
ў
:gradients/softmax_cross_entropy_loss/num_present_grad/TileTile=gradients/softmax_cross_entropy_loss/num_present_grad/Reshape;gradients/softmax_cross_entropy_loss/num_present_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*

Tmultiples0

3gradients/softmax_cross_entropy_loss/Mul_grad/ShapeShape$softmax_cross_entropy_loss/Reshape_2*
T0*
out_type0*
_output_shapes
:
x
5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Cgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/softmax_cross_entropy_loss/Mul_grad/Shape5gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Т
1gradients/softmax_cross_entropy_loss/Mul_grad/mulMul2gradients/softmax_cross_entropy_loss/Sum_grad/Tile&softmax_cross_entropy_loss/ToFloat_1/x*
T0*#
_output_shapes
:џџџџџџџџџ
№
1gradients/softmax_cross_entropy_loss/Mul_grad/SumSum1gradients/softmax_cross_entropy_loss/Mul_grad/mulCgradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ф
5gradients/softmax_cross_entropy_loss/Mul_grad/ReshapeReshape1gradients/softmax_cross_entropy_loss/Mul_grad/Sum3gradients/softmax_cross_entropy_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
Т
3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Mul$softmax_cross_entropy_loss/Reshape_22gradients/softmax_cross_entropy_loss/Sum_grad/Tile*
T0*#
_output_shapes
:џџџџџџџџџ
і
3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_1Sum3gradients/softmax_cross_entropy_loss/Mul_grad/mul_1Egradients/softmax_cross_entropy_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
н
7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1Reshape3gradients/softmax_cross_entropy_loss/Mul_grad/Sum_15gradients/softmax_cross_entropy_loss/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
И
>gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_depsNoOp6^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape8^gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1
Т
Fgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependencyIdentity5gradients/softmax_cross_entropy_loss/Mul_grad/Reshape?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape*#
_output_shapes
:џџџџџџџџџ
Л
Hgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency_1Identity7gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1?^gradients/softmax_cross_entropy_loss/Mul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_loss/Mul_grad/Reshape_1*
_output_shapes
: 

Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
б
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1ShapeBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
out_type0*
_output_shapes
:*
T0
г
]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ShapeOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mulMul:gradients/softmax_cross_entropy_loss/num_present_grad/TileBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*#
_output_shapes
:џџџџџџџџџ
О
Kgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumSumKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul]gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ѕ
Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeReshapeKgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape*
Tshape0*
_output_shapes
: *
T0
э
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1Mul-softmax_cross_entropy_loss/num_present/Select:gradients/softmax_cross_entropy_loss/num_present_grad/Tile*
T0*#
_output_shapes
:џџџџџџџџџ
Ф
Mgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1SumMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/mul_1_gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
И
Qgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1ReshapeMgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Sum_1Ogradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

Xgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_depsNoOpP^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeR^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1

`gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityOgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/ReshapeY^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*b
_classX
VTloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: *
T0
А
bgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityQgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1Y^gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/Reshape_1*#
_output_shapes
:џџџџџџџџџ
Ё
Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
з
Ugradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/SumSumbgradients/softmax_cross_entropy_loss/num_present/broadcast_weights_grad/tuple/control_dependency_1Wgradients/softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

9gradients/softmax_cross_entropy_loss/Reshape_2_grad/ShapeShape#softmax_cross_entropy_loss/xentropy*
out_type0*
_output_shapes
:*
T0

;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeReshapeFgradients/softmax_cross_entropy_loss/Mul_grad/tuple/control_dependency9gradients/softmax_cross_entropy_loss/Reshape_2_grad/Shape*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/zeros_like	ZerosLike%softmax_cross_entropy_loss/xentropy:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Agradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims
ExpandDims;gradients/softmax_cross_entropy_loss/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:џџџџџџџџџ*
T0
о
6gradients/softmax_cross_entropy_loss/xentropy_grad/mulMul=gradients/softmax_cross_entropy_loss/xentropy_grad/ExpandDims%softmax_cross_entropy_loss/xentropy:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

7gradients/softmax_cross_entropy_loss/Reshape_grad/ShapeShapedense_3/BiasAdd*
T0*
out_type0*
_output_shapes
:
ѕ
9gradients/softmax_cross_entropy_loss/Reshape_grad/ReshapeReshape6gradients/softmax_cross_entropy_loss/xentropy_grad/mul7gradients/softmax_cross_entropy_loss/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
А
*gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
 
/gradients/dense_3/BiasAdd_grad/tuple/group_depsNoOp:^gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape+^gradients/dense_3/BiasAdd_grad/BiasAddGrad
А
7gradients/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape0^gradients/dense_3/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_loss/Reshape_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

9gradients/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_3/BiasAdd_grad/BiasAddGrad0^gradients/dense_3/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
д
$gradients/dense_3/MatMul_grad/MatMulMatMul7gradients/dense_3/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ *
transpose_a( 
Ц
&gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu7gradients/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

: *
transpose_a(

.gradients/dense_3/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_3/MatMul_grad/MatMul'^gradients/dense_3/MatMul_grad/MatMul_1

6gradients/dense_3/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_3/MatMul_grad/MatMul/^gradients/dense_3/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_3/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ 

8gradients/dense_3/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_3/MatMul_grad/MatMul_1/^gradients/dense_3/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

: *
T0
Ј
$gradients/dense_2/Relu_grad/ReluGradReluGrad6gradients/dense_3/MatMul_grad/tuple/control_dependencydense_2/Relu*'
_output_shapes
:џџџџџџџџџ *
T0

*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp%^gradients/dense_2/Relu_grad/ReluGrad+^gradients/dense_2/BiasAdd_grad/BiasAddGrad

7gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity$gradients/dense_2/Relu_grad/ReluGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense_2/Relu_grad/ReluGrad*'
_output_shapes
:џџџџџџџџџ *
T0

9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
д
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ *
transpose_a( 
Ф
&gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/Relu7gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:  *
transpose_a(

.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1

6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ 

8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:  *
T0
Є
"gradients/dense/Relu_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu*'
_output_shapes
:џџџџџџџџџ *
T0

(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp#^gradients/dense/Relu_grad/ReluGrad)^gradients/dense/BiasAdd_grad/BiasAddGrad
ў
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/dense/Relu_grad/ReluGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dense/Relu_grad/ReluGrad*'
_output_shapes
:џџџџџџџџџ *
T0
џ
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
Ю
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*'
_output_shapes
:џџџџџџџџџ	*
transpose_a( *
T0
С
$gradients/dense/MatMul_grad/MatMul_1MatMulPlaceholder5gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:	 *
transpose_a(

,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
ќ
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ	*
T0
љ
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes

:	 *
T0

beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0

beta1_power
VariableV2*
_class
loc:@dense/kernel*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
Џ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: 
k
beta1_power/readIdentitybeta1_power*
_class
loc:@dense/kernel*
_output_shapes
: *
T0

beta2_power/initial_valueConst*
valueB
 *wО?*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
_class
loc:@dense/kernel*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
Џ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
k
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 

#dense/kernel/Adam/Initializer/ConstConst*
_class
loc:@dense/kernel*
valueB	 *    *
dtype0*
_output_shapes

:	 
І
dense/kernel/Adam
VariableV2*
	container *
dtype0*
_class
loc:@dense/kernel*
_output_shapes

:	 *
shape
:	 *
shared_name 
Э
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:	 

dense/kernel/Adam/readIdentitydense/kernel/Adam*
_class
loc:@dense/kernel*
_output_shapes

:	 *
T0

%dense/kernel/Adam_1/Initializer/ConstConst*
_class
loc:@dense/kernel*
valueB	 *    *
dtype0*
_output_shapes

:	 
Ј
dense/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape
:	 *
dtype0*
_output_shapes

:	 
г
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/Const*
_class
loc:@dense/kernel*
_output_shapes

:	 *
T0*
validate_shape(*
use_locking(

dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel*
_output_shapes

:	 

!dense/bias/Adam/Initializer/ConstConst*
_class
loc:@dense/bias*
valueB *    *
dtype0*
_output_shapes
: 

dense/bias/Adam
VariableV2*
shape: *
_output_shapes
: *
shared_name *
_class
loc:@dense/bias*
dtype0*
	container 
С
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
u
dense/bias/Adam/readIdentitydense/bias/Adam*
_class
loc:@dense/bias*
_output_shapes
: *
T0

#dense/bias/Adam_1/Initializer/ConstConst*
_class
loc:@dense/bias*
valueB *    *
dtype0*
_output_shapes
: 

dense/bias/Adam_1
VariableV2*
_class
loc:@dense/bias*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
Ч
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
y
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
_class
loc:@dense/bias*
_output_shapes
: *
T0

%dense_1/kernel/Adam/Initializer/ConstConst*!
_class
loc:@dense_1/kernel*
valueB  *    *
_output_shapes

:  *
dtype0
Њ
dense_1/kernel/Adam
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape
:  *
dtype0*
_output_shapes

:  
е
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adam%dense_1/kernel/Adam/Initializer/Const*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:  

dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0

'dense_1/kernel/Adam_1/Initializer/ConstConst*!
_class
loc:@dense_1/kernel*
valueB  *    *
_output_shapes

:  *
dtype0
Ќ
dense_1/kernel/Adam_1
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape
:  *
dtype0*
_output_shapes

:  
л
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1'dense_1/kernel/Adam_1/Initializer/Const*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:  

dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0

#dense_1/bias/Adam/Initializer/ConstConst*
_class
loc:@dense_1/bias*
valueB *    *
_output_shapes
: *
dtype0

dense_1/bias/Adam
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes
: *
shape: *
dtype0*
shared_name *
	container 
Щ
dense_1/bias/Adam/AssignAssigndense_1/bias/Adam#dense_1/bias/Adam/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
{
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
_class
loc:@dense_1/bias*
_output_shapes
: *
T0

%dense_1/bias/Adam_1/Initializer/ConstConst*
_class
loc:@dense_1/bias*
valueB *    *
dtype0*
_output_shapes
: 
 
dense_1/bias/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
: *
shape: *
shared_name 
Я
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1%dense_1/bias/Adam_1/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 

dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
: 

%dense_2/kernel/Adam/Initializer/ConstConst*!
_class
loc:@dense_2/kernel*
valueB *    *
dtype0*
_output_shapes

: 
Њ
dense_2/kernel/Adam
VariableV2*
shape
: *
_output_shapes

: *
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container 
е
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adam%dense_2/kernel/Adam/Initializer/Const*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(

dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0

'dense_2/kernel/Adam_1/Initializer/ConstConst*!
_class
loc:@dense_2/kernel*
valueB *    *
dtype0*
_output_shapes

: 
Ќ
dense_2/kernel/Adam_1
VariableV2*
shape
: *
_output_shapes

: *
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container 
л
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1'dense_2/kernel/Adam_1/Initializer/Const*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 

dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 

#dense_2/bias/Adam/Initializer/ConstConst*
_class
loc:@dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:

dense_2/bias/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container 
Щ
dense_2/bias/Adam/AssignAssigndense_2/bias/Adam#dense_2/bias/Adam/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
{
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0

%dense_2/bias/Adam_1/Initializer/ConstConst*
_class
loc:@dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
 
dense_2/bias/Adam_1
VariableV2*
	container *
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
:*
shape:*
shared_name 
Я
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1%dense_2/bias/Adam_1/Initializer/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wО?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
и
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
_output_shapes

:	 
Ы
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
_output_shapes
: *
T0*
use_locking( 
ф
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:  
з
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
: *
T0*
use_locking( 
ф
$Adam/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients/dense_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

: 
з
"Adam/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/dense_3/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0*
use_locking( 
Я
Adam/mulMulbeta1_power/read
Adam/beta1#^Adam/update_dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
б

Adam/mul_1Mulbeta2_power/read
Adam/beta2#^Adam/update_dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam*
_class
loc:@dense/kernel*
_output_shapes
: *
T0

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 

AdamNoOp#^Adam/update_dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_2/kernel/ApplyAdam#^Adam/update_dense_2/bias/ApplyAdam^Adam/Assign^Adam/Assign_1

initNoOp^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^beta1_power/Assign^beta2_power/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense_1/kernel/Adam/Assign^dense_1/kernel/Adam_1/Assign^dense_1/bias/Adam/Assign^dense_1/bias/Adam_1/Assign^dense_2/kernel/Adam/Assign^dense_2/kernel/Adam_1/Assign^dense_2/bias/Adam/Assign^dense_2/bias/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0
Ц
save/SaveV2/tensor_namesConst*љ
valueяBьBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bdense_2/biasBdense_2/bias/AdamBdense_2/bias/Adam_1Bdense_2/kernelBdense_2/kernel/AdamBdense_2/kernel/Adam_1*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
с
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1dense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1dense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1*"
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
_output_shapes
: *
T0
o
save/RestoreV2/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignbeta1_powersave/RestoreV2*
_class
loc:@dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
q
save/RestoreV2_1/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ё
save/Assign_1Assignbeta2_powersave/RestoreV2_1*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: 
p
save/RestoreV2_2/tensor_namesConst*
valueBB
dense/bias*
_output_shapes
:*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
Ђ
save/Assign_2Assign
dense/biassave/RestoreV2_2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
u
save/RestoreV2_3/tensor_namesConst*$
valueBBdense/bias/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
Ї
save/Assign_3Assigndense/bias/Adamsave/RestoreV2_3*
_class
loc:@dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
w
save/RestoreV2_4/tensor_namesConst*&
valueBBdense/bias/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Љ
save/Assign_4Assigndense/bias/Adam_1save/RestoreV2_4*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
r
save/RestoreV2_5/tensor_namesConst*!
valueBBdense/kernel*
_output_shapes
:*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
Њ
save/Assign_5Assigndense/kernelsave/RestoreV2_5*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:	 
w
save/RestoreV2_6/tensor_namesConst*&
valueBBdense/kernel/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
Џ
save/Assign_6Assigndense/kernel/Adamsave/RestoreV2_6*
_class
loc:@dense/kernel*
_output_shapes

:	 *
T0*
validate_shape(*
use_locking(
y
save/RestoreV2_7/tensor_namesConst*(
valueBBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Б
save/Assign_7Assigndense/kernel/Adam_1save/RestoreV2_7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:	 
r
save/RestoreV2_8/tensor_namesConst*!
valueBBdense_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
І
save/Assign_8Assigndense_1/biassave/RestoreV2_8*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
w
save/RestoreV2_9/tensor_namesConst*&
valueBBdense_1/bias/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ћ
save/Assign_9Assigndense_1/bias/Adamsave/RestoreV2_9*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
z
save/RestoreV2_10/tensor_namesConst*(
valueBBdense_1/bias/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Џ
save/Assign_10Assigndense_1/bias/Adam_1save/RestoreV2_10*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
u
save/RestoreV2_11/tensor_namesConst*#
valueBBdense_1/kernel*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
А
save/Assign_11Assigndense_1/kernelsave/RestoreV2_11*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:  
z
save/RestoreV2_12/tensor_namesConst*(
valueBBdense_1/kernel/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
Е
save/Assign_12Assigndense_1/kernel/Adamsave/RestoreV2_12*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:  
|
save/RestoreV2_13/tensor_namesConst**
value!BBdense_1/kernel/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
З
save/Assign_13Assigndense_1/kernel/Adam_1save/RestoreV2_13*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0*
validate_shape(*
use_locking(
s
save/RestoreV2_14/tensor_namesConst*!
valueBBdense_2/bias*
_output_shapes
:*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ј
save/Assign_14Assigndense_2/biassave/RestoreV2_14*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
x
save/RestoreV2_15/tensor_namesConst*&
valueBBdense_2/bias/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save/Assign_15Assigndense_2/bias/Adamsave/RestoreV2_15*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
z
save/RestoreV2_16/tensor_namesConst*(
valueBBdense_2/bias/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
Џ
save/Assign_16Assigndense_2/bias/Adam_1save/RestoreV2_16*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
u
save/RestoreV2_17/tensor_namesConst*#
valueBBdense_2/kernel*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
А
save/Assign_17Assigndense_2/kernelsave/RestoreV2_17*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
z
save/RestoreV2_18/tensor_namesConst*(
valueBBdense_2/kernel/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
Е
save/Assign_18Assigndense_2/kernel/Adamsave/RestoreV2_18*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
|
save/RestoreV2_19/tensor_namesConst**
value!BBdense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
З
save/Assign_19Assigndense_2/kernel/Adam_1save/RestoreV2_19*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 
р
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19
Q
keyPlaceholder*#
_output_shapes
:џџџџџџџџџ*
shape: *
dtype0	
G
IdentityIdentitykey*
T0	*#
_output_shapes
:џџџџџџџџџ
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_76564a99db3d4527a099575997e70766/part*
_output_shapes
: *
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Ш
save_1/SaveV2/tensor_namesConst*љ
valueяBьBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bdense_2/biasBdense_2/bias/AdamBdense_2/bias/Adam_1Bdense_2/kernelBdense_2/kernel/AdamBdense_2/kernel/Adam_1*
dtype0*
_output_shapes
:

save_1/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ѓ
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1dense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1dense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1*"
dtypes
2

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
Ѓ
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
_output_shapes
:*
T0*
N

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
_output_shapes
: *
T0
q
save_1/RestoreV2/tensor_namesConst* 
valueBBbeta1_power*
_output_shapes
:*
dtype0
j
!save_1/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
Ё
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
_class
loc:@dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
s
save_1/RestoreV2_1/tensor_namesConst* 
valueBBbeta2_power*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ѕ
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2_1*
_class
loc:@dense/kernel*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
r
save_1/RestoreV2_2/tensor_namesConst*
valueBB
dense/bias*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
І
save_1/Assign_2Assign
dense/biassave_1/RestoreV2_2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
w
save_1/RestoreV2_3/tensor_namesConst*$
valueBBdense/bias/Adam*
_output_shapes
:*
dtype0
l
#save_1/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
Ћ
save_1/Assign_3Assigndense/bias/Adamsave_1/RestoreV2_3*
_class
loc:@dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
y
save_1/RestoreV2_4/tensor_namesConst*&
valueBBdense/bias/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save_1/Assign_4Assigndense/bias/Adam_1save_1/RestoreV2_4*
_class
loc:@dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
t
save_1/RestoreV2_5/tensor_namesConst*!
valueBBdense/kernel*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
Ў
save_1/Assign_5Assigndense/kernelsave_1/RestoreV2_5*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:	 
y
save_1/RestoreV2_6/tensor_namesConst*&
valueBBdense/kernel/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Г
save_1/Assign_6Assigndense/kernel/Adamsave_1/RestoreV2_6*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:	 
{
save_1/RestoreV2_7/tensor_namesConst*(
valueBBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
Е
save_1/Assign_7Assigndense/kernel/Adam_1save_1/RestoreV2_7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:	 
t
save_1/RestoreV2_8/tensor_namesConst*!
valueBBdense_1/bias*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Њ
save_1/Assign_8Assigndense_1/biassave_1/RestoreV2_8*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
: 
y
save_1/RestoreV2_9/tensor_namesConst*&
valueBBdense_1/bias/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_9/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0

save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
Џ
save_1/Assign_9Assigndense_1/bias/Adamsave_1/RestoreV2_9*
_class
loc:@dense_1/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
|
 save_1/RestoreV2_10/tensor_namesConst*(
valueBBdense_1/bias/Adam_1*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_10/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ё
save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Г
save_1/Assign_10Assigndense_1/bias/Adam_1save_1/RestoreV2_10*
_class
loc:@dense_1/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
w
 save_1/RestoreV2_11/tensor_namesConst*#
valueBBdense_1/kernel*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_11/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ё
save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
Д
save_1/Assign_11Assigndense_1/kernelsave_1/RestoreV2_11*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0*
validate_shape(*
use_locking(
|
 save_1/RestoreV2_12/tensor_namesConst*(
valueBBdense_1/kernel/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_12/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ё
save_1/RestoreV2_12	RestoreV2save_1/Const save_1/RestoreV2_12/tensor_names$save_1/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
Й
save_1/Assign_12Assigndense_1/kernel/Adamsave_1/RestoreV2_12*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:  
~
 save_1/RestoreV2_13/tensor_namesConst**
value!BBdense_1/kernel/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ё
save_1/RestoreV2_13	RestoreV2save_1/Const save_1/RestoreV2_13/tensor_names$save_1/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
Л
save_1/Assign_13Assigndense_1/kernel/Adam_1save_1/RestoreV2_13*!
_class
loc:@dense_1/kernel*
_output_shapes

:  *
T0*
validate_shape(*
use_locking(
u
 save_1/RestoreV2_14/tensor_namesConst*!
valueBBdense_2/bias*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ё
save_1/RestoreV2_14	RestoreV2save_1/Const save_1/RestoreV2_14/tensor_names$save_1/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ќ
save_1/Assign_14Assigndense_2/biassave_1/RestoreV2_14*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
z
 save_1/RestoreV2_15/tensor_namesConst*&
valueBBdense_2/bias/Adam*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ё
save_1/RestoreV2_15	RestoreV2save_1/Const save_1/RestoreV2_15/tensor_names$save_1/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Б
save_1/Assign_15Assigndense_2/bias/Adamsave_1/RestoreV2_15*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
|
 save_1/RestoreV2_16/tensor_namesConst*(
valueBBdense_2/bias/Adam_1*
_output_shapes
:*
dtype0
m
$save_1/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ё
save_1/RestoreV2_16	RestoreV2save_1/Const save_1/RestoreV2_16/tensor_names$save_1/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
Г
save_1/Assign_16Assigndense_2/bias/Adam_1save_1/RestoreV2_16*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
w
 save_1/RestoreV2_17/tensor_namesConst*#
valueBBdense_2/kernel*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_17/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ё
save_1/RestoreV2_17	RestoreV2save_1/Const save_1/RestoreV2_17/tensor_names$save_1/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Д
save_1/Assign_17Assigndense_2/kernelsave_1/RestoreV2_17*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
|
 save_1/RestoreV2_18/tensor_namesConst*(
valueBBdense_2/kernel/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_18/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ё
save_1/RestoreV2_18	RestoreV2save_1/Const save_1/RestoreV2_18/tensor_names$save_1/RestoreV2_18/shape_and_slices*
_output_shapes
:*
dtypes
2
Й
save_1/Assign_18Assigndense_2/kernel/Adamsave_1/RestoreV2_18*!
_class
loc:@dense_2/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
~
 save_1/RestoreV2_19/tensor_namesConst**
value!BBdense_2/kernel/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_19/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
Ё
save_1/RestoreV2_19	RestoreV2save_1/Const save_1/RestoreV2_19/tensor_names$save_1/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
Л
save_1/Assign_19Assigndense_2/kernel/Adam_1save_1/RestoreV2_19*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

: 

save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"Ѕ
	variables
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
I
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:0
O
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:0
C
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:0
I
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:0
O
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assigndense_1/kernel/Adam/read:0
U
dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assigndense_1/kernel/Adam_1/read:0
I
dense_1/bias/Adam:0dense_1/bias/Adam/Assigndense_1/bias/Adam/read:0
O
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assigndense_1/bias/Adam_1/read:0
O
dense_2/kernel/Adam:0dense_2/kernel/Adam/Assigndense_2/kernel/Adam/read:0
U
dense_2/kernel/Adam_1:0dense_2/kernel/Adam_1/Assigndense_2/kernel/Adam_1/read:0
I
dense_2/bias/Adam:0dense_2/bias/Adam/Assigndense_2/bias/Adam/read:0
O
dense_2/bias/Adam_1:0dense_2/bias/Adam_1/Assigndense_2/bias/Adam_1/read:0"
trainable_variablesёю
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0"
train_op

Adam"0
losses&
$
"softmax_cross_entropy_loss/value:0*Щ
serving_defaultЕ
)
x$
Placeholder:0џџџџџџџџџ	

key
key:0	џџџџџџџџџ%
y 
	Softmax:0џџџџџџџџџ$
key

Identity:0	џџџџџџџџџtensorflow/serving/predict