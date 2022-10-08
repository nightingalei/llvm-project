//===- SPIRVCooperativeTensorOps.td - cooperative operations ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of cooperative tensor extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_COOPERATIVE_TENSOR_OPS
#define SPIRV_COOPERATIVE_TENSOR_OPS

// -----

def SPIRV_VSICooperativeTensorLengthOp : SPIRV_VsiVendorOp<"CooperativeTensorLength",
  [NoSideEffect]> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Number of components of a cooperative tensor type accessible to each
    invocation when treated as a composite.

    Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

    Type is a cooperative tensor type.

    ``` {.ebnf}
    cooperative-tensor-length-op ::= ssa-id `=` `spirv.VSI.CooperativeTensorLength
                                    ` : ` cooperative-tensor-type
    ```

    For example:

    ```
    %0 = spirv.VSI.CooperativeTensorLength : !spirv.cooptensor<i32, 8, 16>
    ```
  }];

  let assemblyFormat = "attr-dict `:` $cooperative_tensor_type";

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPIRV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    TypeAttr:$cooperative_tensor_type
  );

  let results = (outs
    SPIRV_Int32:$result
  );
  let hasVerifier = 0;
}

// -----

def SPIRV_VSICooperativeTensorLoadOp : SPIRV_VsiVendorOp<"CooperativeTensorLoad", []> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Load a cooperative tensor through a pointer.

    Result Type is the type of the loaded object. It must be a cooperative
    tensor type.

    Pointer is a pointer into an array. Its type must be an OpTypePointer whose
    Type operand is a scalar or vector type. The storage class of Pointer must
    be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
    supported) PhysicalStorageBufferEXT.

    For a given dynamic instance of this instruction, all operands of this
    instruction must be the same for all invocations in a given scope instance
    (where the scope is the scope the cooperative tensor type was created with).
    All invocations in a given scope instance must be active or all must be
    inactive.

    ### Custom assembly form

    ``` {.ebnf}
    cooperative-tensorload-op ::= ssa-id `=` `spirv.VSI.CooperativeTensorLoad`
                              ssa-use `,` ssa-use `,` ssa-use
                              (`[` memory-access `]`)? ` : `
                              pointer-type `as`
                              cooperative-tensor-type
    ```

    For example:

    ```
    %0 = spirv.VSI.CooperativeTensorLoad %ptr, %offset, %strides, %hwcLayout
         : !spirv.ptr<i32, StorageBuffer> as !spirv.cooptensor<16x8xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPIRV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_Integer:$offset,
    SPIRV_Vector:$strides,
    SPIRV_Bool:$hwclayout,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs
    SPIRV_AnyCooperativeTensor:$result
  );

  let hasVerifier = 0;
  //let verifier = [{
  //  return verifyPointerAndCoopTensorType(*this, pointer().getType(),
  //                                        result().getType());
  //}];
}

// -----

def SPIRV_VSICooperativeTensorMatMulAddOp : SPIRV_VsiVendorOp<"CooperativeTensorMatMulAdd",
  [NoSideEffect, AllTypesMatch<["c", "result"]>]> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Linear-algebraic tensor multiply of A by B and then component-wise add C.
    The order of the operations is implementation-dependent. The internal
    precision of floating-point operations is defined by the client API.
    Integer operations are performed at the precision of the Result Type and are
    exact unless there is overflow or underflow, in which case the result is
    undefined.

    Result Type must be a cooperative tensor type with M rows and N columns.

    A is a cooperative tensor with M rows and K columns.

    B is a cooperative tensor with K rows and N columns.

    C is a cooperative tensor with M rows and N columns.

    The values of M, N, and K must be consistent across the result and operands.
    This is referred to as an MxNxK tensor multiply.

    A, B, C, and Result Type must have the same scope, and this defines the
    scope of the operation. A, B, C, and Result Type need not necessarily have
    the same component type, this is defined by the client API.

    If the Component Type of any tensor operand is an integer type, then its
    components are treated as signed if its Component Type has Signedness of 1
    and are treated as unsigned otherwise.

    For a given dynamic instance of this instruction, all invocations in a given
    scope instance must be active or all must be inactive (where the scope is
    the scope of the operation).

    ``` {.ebnf}
    cooperative-tensormatmuladd-op ::= ssa-id `=` `spirv.VSI.CooperativeTensorMatMulAdd`
                              ssa-use `,` ssa-use `,` ssa-use ` : `
                              a-cooperative-tensor-type,
                              b-cooperative-tensor-type ->
                              result-cooperative-tensor-type
    ```
    For example:

    ```
    %0 = spirv.VSI.CooperativeTensorMatMulAdd %arg0, %arg1, %arg2,  :
      !spirv.cooptensor<i32, 8, 16>
    ```
  }];

  let assemblyFormat = [{
    operands attr-dict`:` type($a) `,` type($b) `->` type($c)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPIRV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPIRV_AnyCooperativeTensor:$a,
    SPIRV_AnyCooperativeTensor:$b,
    SPIRV_AnyCooperativeTensor:$c
  );

  let results = (outs
    SPIRV_AnyCooperativeTensor:$result
  );

  let hasVerifier = 0;
  //let verifier = [{ return verifyCoopTensorMatMulAdd(*this); }];
}

// -----

def SPIRV_VSICooperativeTensorStoreOp : SPIRV_VsiVendorOp<"CooperativeTensorStore", []> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Store a cooperative tensor through a pointer.

    Pointer is a pointer into an array. Its type must be an OpTypePointer whose
    Type operand is a scalar or vector type. The storage class of Pointer must
    be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
    supported) PhysicalStorageBufferEXT.

    Object is the object to store. Its type must be an
    OpTypeCooperativeTensorVSI.

    Memory Access must be a Memory Access literal. If not present, it is the
    same as specifying None.

    ``` {.ebnf}
    coop-tensor-store-op ::= `spirv.VSI.CooperativeTensorStore `
                              ssa-use `, ` ssa-use `, `
                              ssa-use `, ` ssa-use `, `
                              (`[` memory-access `]`)? `:`
                              pointer-type `,` spirv-element-type
    ```

    For example:

    ```
      spirv.VSI.CooperativeTensorStore %arg0, %arg2, %arg1, %arg3 :
        !spirv.ptr<i32, StorageBuffer>, !spirv.cooptensor<16x8xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPIRV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_AnyCooperativeTensor:$object,
    SPIRV_Integer:$offset,
    SPIRV_Vector:$strides,
    SPIRV_Bool:$hwclayout,
    OptionalAttr<SPIRV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs);

  let hasVerifier = 0;
  //let verifier = [{
  //  return verifyPointerAndCoopTensorType(*this, pointer().getType(),
  //                                        object().getType());
  //}];
}

// -----

def SPIRV_VSICooperativeTensorConv2DOp : SPIRV_VsiVendorOp<"CooperativeTensorConv2D",
  [NoSideEffect]> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    ``` {.ebnf}
    cooperative-tensormatmuladd-op ::= ssa-id `=` `spirv.VSI.CooperativeTensorConv2D`
                              ssa-use `,` ssa-use `,` ssa-use `,` ssa-use ` : `
                              input-cooperative-tensor-type,
                              weight-cooperative-tensor-type,
                              bias-cooperative-tensor-type,
                              strides-vector-type ->
                              result-cooperative-tensor-type
    ```
    For example:

    ```
    %0 = spirv.VSI.CooperativeTensorConv2D %input, %weight, %bias, %strides  :
      !spirv.cooptensor<i32, 8, 16>
    ```
  }];

  let assemblyFormat = [{
    operands attr-dict`:` type($input) `,` type($weight) `,` type($bias) `,` type($strides) `->` type($result)
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPIRV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPIRV_AnyCooperativeTensor:$input,
    SPIRV_AnyCooperativeTensor:$weight,
    SPIRV_AnyCooperativeTensor:$bias,
    SPIRV_Vector:$strides
  );

  let results = (outs
    SPIRV_AnyCooperativeTensor:$result
  );

  let hasVerifier = 0;
  //let verifier = [{ return verifyCoopTensorConv2D(*this); }];
}

// -----

#endif // SPIRV_COOPERATIVE_TENSOR_OPS
