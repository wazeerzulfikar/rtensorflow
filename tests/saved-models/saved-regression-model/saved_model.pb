˝|
Ă
9
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
8
Const
output"dtype"
valuetensor"
dtypetype
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	
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
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
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
0
Square
x"T
y"T"
Ttype:
	2	
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
shared_namestring "train"serve*1.2.12v1.2.0-5-g435cdfcf
d
xPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
yPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
random_uniform/shapeConst*
dtype0*
valueB:*
_output_shapes
:
W
random_uniform/minConst*
dtype0*
valueB
 *  ż*
_output_shapes
: 
W
random_uniform/maxConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

random_uniform/RandomUniformRandomUniformrandom_uniform/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
b
random_uniform/subSubrandom_uniform/maxrandom_uniform/min*
T0*
_output_shapes
: 
p
random_uniform/mulMulrandom_uniform/RandomUniformrandom_uniform/sub*
T0*
_output_shapes
:
b
random_uniformAddrandom_uniform/mulrandom_uniform/min*
T0*
_output_shapes
:
m
w
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:

w/AssignAssignwrandom_uniform*
validate_shape(*
_class

loc:@w*
use_locking(*
T0*
_output_shapes
:
P
w/readIdentityw*
_class

loc:@w*
T0*
_output_shapes
:
R
zerosConst*
dtype0*
valueB*    *
_output_shapes
:
m
b
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:

b/AssignAssignbzeros*
validate_shape(*
_class

loc:@b*
use_locking(*
T0*
_output_shapes
:
P
b/readIdentityb*
_class

loc:@b*
T0*
_output_shapes
:
G
mulMulw/readx*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
y_hatAddmulb/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
F
subSuby_haty*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
G
SquareSquaresub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
ConstConst*
dtype0*
valueB"       *
_output_shapes
:
Y
lossMeanSquareConst*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
r
!gradients/loss_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:

gradients/loss_grad/ReshapeReshapegradients/Fill!gradients/loss_grad/Reshape/shape*
_output_shapes

:*
Tshape0*
T0
_
gradients/loss_grad/ShapeShapeSquare*
out_type0*
T0*
_output_shapes
:

gradients/loss_grad/TileTilegradients/loss_grad/Reshapegradients/loss_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients/loss_grad/Shape_1ShapeSquare*
out_type0*
T0*
_output_shapes
:
^
gradients/loss_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
c
gradients/loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:

gradients/loss_grad/ProdProdgradients/loss_grad/Shape_1gradients/loss_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
e
gradients/loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:

gradients/loss_grad/Prod_1Prodgradients/loss_grad/Shape_2gradients/loss_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
_
gradients/loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 

gradients/loss_grad/MaximumMaximumgradients/loss_grad/Prod_1gradients/loss_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/loss_grad/floordivFloorDivgradients/loss_grad/Prodgradients/loss_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/loss_grad/CastCastgradients/loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/loss_grad/truedivRealDivgradients/loss_grad/Tilegradients/loss_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
gradients/Square_grad/mul/xConst^gradients/loss_grad/truediv*
dtype0*
valueB
 *   @*
_output_shapes
: 
t
gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Square_grad/mul_1Mulgradients/loss_grad/truedivgradients/Square_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
gradients/sub_grad/ShapeShapey_hat*
out_type0*
T0*
_output_shapes
:
[
gradients/sub_grad/Shape_1Shapey*
out_type0*
T0*
_output_shapes
:
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¤
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
¨
gradients/sub_grad/Sum_1Sumgradients/Square_grad/mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ú
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
gradients/y_hat_grad/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
f
gradients/y_hat_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
ş
*gradients/y_hat_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/y_hat_grad/Shapegradients/y_hat_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¸
gradients/y_hat_grad/SumSum+gradients/sub_grad/tuple/control_dependency*gradients/y_hat_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/y_hat_grad/ReshapeReshapegradients/y_hat_grad/Sumgradients/y_hat_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
ź
gradients/y_hat_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency,gradients/y_hat_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/y_hat_grad/Reshape_1Reshapegradients/y_hat_grad/Sum_1gradients/y_hat_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/y_hat_grad/tuple/group_depsNoOp^gradients/y_hat_grad/Reshape^gradients/y_hat_grad/Reshape_1
â
-gradients/y_hat_grad/tuple/control_dependencyIdentitygradients/y_hat_grad/Reshape&^gradients/y_hat_grad/tuple/group_deps*/
_class%
#!loc:@gradients/y_hat_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
/gradients/y_hat_grad/tuple/control_dependency_1Identitygradients/y_hat_grad/Reshape_1&^gradients/y_hat_grad/tuple/group_deps*1
_class'
%#loc:@gradients/y_hat_grad/Reshape_1*
T0*
_output_shapes
:
b
gradients/mul_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
[
gradients/mul_grad/Shape_1Shapex*
out_type0*
T0*
_output_shapes
:
´
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mulMul-gradients/y_hat_grad/tuple/control_dependencyx*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
_output_shapes
:*
Tshape0*
T0

gradients/mul_grad/mul_1Mulw/read-gradients/y_hat_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
Í
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*
_output_shapes
:
ŕ
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
train/learning_rateConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Î
#train/update_w/ApplyGradientDescentApplyGradientDescentwtrain/learning_rate+gradients/mul_grad/tuple/control_dependency*
_class

loc:@w*
use_locking( *
T0*
_output_shapes
:
Ň
#train/update_b/ApplyGradientDescentApplyGradientDescentbtrain/learning_rate/gradients/y_hat_grad/tuple/control_dependency_1*
_class

loc:@b*
use_locking( *
T0*
_output_shapes
:
Y
trainNoOp$^train/update_w/ApplyGradientDescent$^train/update_b/ApplyGradientDescent
"
initNoOp	^w/Assign	^b/Assign
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_7094d1ec5aec4c31a1d256368ef0fb05/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
e
save/SaveV2/tensor_namesConst*
dtype0*
valueBBbBw*
_output_shapes
:
g
save/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B *
_output_shapes
:
{
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbw*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*

axis *
T0*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
e
save/RestoreV2/tensor_namesConst*
dtype0*
valueBBb*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignbsave/RestoreV2*
validate_shape(*
_class

loc:@b*
use_locking(*
T0*
_output_shapes
:
g
save/RestoreV2_1/tensor_namesConst*
dtype0*
valueBBw*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_1Assignwsave/RestoreV2_1*
validate_shape(*
_class

loc:@w*
use_locking(*
T0*
_output_shapes
:
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
train_op	

train"E
	variables86

w:0w/Assignw/read:0

b:0b/Assignb/read:0"O
trainable_variables86

w:0w/Assignw/read:0

b:0b/Assignb/read:0