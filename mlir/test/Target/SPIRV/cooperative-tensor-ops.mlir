// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [CooperativeTensorVSI], [SPV_VSI_cooperative_tensor]> {
  // CHECK-LABEL: @cooperative_tensor_load
  spv.func @cooperative_tensor_load(%ptr : !spv.ptr<i32, StorageBuffer>, %offset: i32, %stride : i32, %b : i1) "None" {
    // CHECK: {{%.*}} = spv.CooperativeTensorLoadVSI {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : !spv.ptr<i32, StorageBuffer> as !spv.cooptensor<3x3x128xi32>
    %0 = spv.CooperativeTensorLoadVSI %ptr, %offset, %stride, %b : !spv.ptr<i32, StorageBuffer> as !spv.cooptensor<3x3x128xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_load_memaccess
  spv.func @cooperative_tensor_load_memaccess(%ptr : !spv.ptr<i32, StorageBuffer>, %offset: i32, %stride : i32, %b : i1) "None" {
    // CHECK: {{%.*}} = spv.CooperativeTensorLoadVSI {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spv.ptr<i32, StorageBuffer> as !spv.cooptensor<3x3x128xi32>
    %0 = spv.CooperativeTensorLoadVSI %ptr, %offset, %stride, %b ["Volatile"] : !spv.ptr<i32, StorageBuffer> as !spv.cooptensor<3x3x128xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_store
  spv.func @cooperative_tensor_store(%ptr : !spv.ptr<i32, StorageBuffer>, %offset: i32, %stride : i32, %m : !spv.cooptensor<3x3x128xi32>, %b : i1) "None" {
    // CHECK: spv.CooperativeTensorStoreVSI {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : !spv.ptr<i32, StorageBuffer>, !spv.cooptensor<3x3x128xi32>
    spv.CooperativeTensorStoreVSI %ptr, %m, %offset, %stride, %b : !spv.ptr<i32, StorageBuffer>, !spv.cooptensor<3x3x128xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_store_memaccess
  spv.func @cooperative_tensor_store_memaccess(%ptr : !spv.ptr<i32, StorageBuffer>, %m : !spv.cooptensor<3x3x128xi32>, %offset: i32, %stride : i32, %b : i1) "None" {
    // CHECK: spv.CooperativeTensorStoreVSI {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spv.ptr<i32, StorageBuffer>, !spv.cooptensor<3x3x128xi32>
    spv.CooperativeTensorStoreVSI %ptr, %m, %offset, %stride, %b ["Volatile"] : !spv.ptr<i32, StorageBuffer>, !spv.cooptensor<3x3x128xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_length
  spv.func @cooperative_tensor_length() -> i32 "None" {
    // CHECK: {{%.*}} = spv.CooperativeTensorLengthVSI : !spv.cooptensor<3x3x128xi32>
    %0 = spv.CooperativeTensorLengthVSI : !spv.cooptensor<3x3x128xi32>
    spv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @cooperative_tensor_muladd
  spv.func @cooperative_tensor_muladd(%a : !spv.cooptensor<8x16xi32>, %b : !spv.cooptensor<16x8xi32>, %c : !spv.cooptensor<8x8xi32>) "None" {
    // CHECK: {{%.*}} = spv.CooperativeTensorMatMulAddVSI {{%.*}}, {{%.*}}, {{%.*}}  : !spv.cooptensor<8x16xi32>, !spv.cooptensor<16x8xi32> -> !spv.cooptensor<8x8xi32>
    %r = spv.CooperativeTensorMatMulAddVSI %a, %b, %c : !spv.cooptensor<8x16xi32>, !spv.cooptensor<16x8xi32> -> !spv.cooptensor<8x8xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_add
  spv.func @cooperative_tensor_add(%a : !spv.cooptensor<8x16xi32>, %b : !spv.cooptensor<8x16xi32>) "None" {
    // CHECK: {{%.*}} = spv.IAdd {{%.*}}, {{%.*}} : !spv.cooptensor<8x16xi32>
    %r = spv.IAdd %a, %b : !spv.cooptensor<8x16xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_sub
  spv.func @cooperative_tensor_sub(%a : !spv.cooptensor<8x16xi32>, %b : !spv.cooptensor<8x16xi32>) "None" {
    // CHECK: {{%.*}} = spv.ISub {{%.*}}, {{%.*}} : !spv.cooptensor<8x16xi32>
    %r = spv.ISub %a, %b : !spv.cooptensor<8x16xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_sdiv
  spv.func @cooperative_tensor_sdiv(%a : !spv.cooptensor<8x16xi32>, %b : !spv.cooptensor<8x16xi32>) "None" {
    // CHECK: {{%.*}} = spv.SDiv {{%.*}}, {{%.*}} : !spv.cooptensor<8x16xi32>
    %r = spv.SDiv %a, %b : !spv.cooptensor<8x16xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_udiv
  spv.func @cooperative_tensor_udiv(%a : !spv.cooptensor<8x16xi32>, %b : !spv.cooptensor<8x16xi32>) "None" {
    // CHECK: {{%.*}} = spv.UDiv {{%.*}}, {{%.*}} : !spv.cooptensor<8x16xi32>
    %r = spv.UDiv %a, %b : !spv.cooptensor<8x16xi32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_fadd
  spv.func @cooperative_tensor_fadd(%a : !spv.cooptensor<8x16xf32>, %b : !spv.cooptensor<8x16xf32>) "None" {
    // CHECK: {{%.*}} = spv.FAdd {{%.*}}, {{%.*}} : !spv.cooptensor<8x16xf32>
    %r = spv.FAdd %a, %b : !spv.cooptensor<8x16xf32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_fsub
  spv.func @cooperative_tensor_fsub(%a : !spv.cooptensor<8x16xf32>, %b : !spv.cooptensor<8x16xf32>) "None" {
    // CHECK: {{%.*}} = spv.FSub {{%.*}}, {{%.*}} : !spv.cooptensor<8x16xf32>
    %r = spv.FSub %a, %b : !spv.cooptensor<8x16xf32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_fdiv
  spv.func @cooperative_tensor_fdiv(%a : !spv.cooptensor<8x16xf32>, %b : !spv.cooptensor<8x16xf32>) "None" {
    // CHECK: {{%.*}} = spv.FDiv {{%.*}}, {{%.*}} : !spv.cooptensor<8x16xf32>
    %r = spv.FDiv %a, %b : !spv.cooptensor<8x16xf32>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_tensor_access_chain
  spv.func @cooperative_tensor_access_chain(%a : !spv.ptr<!spv.cooptensor<8x16xf32>, Function>) -> !spv.ptr<f32, Function> "None" {
    %0 = spv.Constant 0: i32
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.cooptensor<8x16xf32>, Function>, i32
    %1 = spv.AccessChain %a[%0] : !spv.ptr<!spv.cooptensor<8x16xf32>, Function>, i32
    spv.ReturnValue %1 : !spv.ptr<f32, Function>
  }
}
