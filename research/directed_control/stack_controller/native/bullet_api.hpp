#pragma once
#include <SharedMemory/PhysicsClientC_API.h>
#include <SharedMemory/PhysicsDirectC_API.h>
#ifdef REFERENCE_ENGINE
#include <dlfcn.h>
#include <stdexcept>
#include <cstdlib>
namespace native_reference {
inline void* engine=[](){
 const char* lib=std::getenv("EXPERT_LIBPYTHON");
 const char* engine=std::getenv("EXPERT_PYBULLET");
 if(!lib)lib=DEFAULT_LIBPYTHON;
 if(!engine)engine=DEFAULT_PYBULLET;
 if(!dlopen(lib,RTLD_LAZY|RTLD_GLOBAL))throw std::runtime_error(dlerror());
 int flags=RTLD_LAZY|RTLD_LOCAL;
#ifdef __linux__
 // Bullet bundles its own XML parser. On ELF, prefer that implementation over
 // incompatible parser symbols already loaded by Pinocchio's dependencies.
 flags|=RTLD_DEEPBIND;
#endif
 void* handle=dlopen(engine,flags);if(!handle)throw std::runtime_error(dlerror());return handle;
}();
inline auto p_b3CollisionFilterCommandInit=reinterpret_cast<decltype(&::b3CollisionFilterCommandInit)>(dlsym(engine,"b3CollisionFilterCommandInit"));
inline auto p_b3SetCollisionFilterGroupMask=reinterpret_cast<decltype(&::b3SetCollisionFilterGroupMask)>(dlsym(engine,"b3SetCollisionFilterGroupMask"));
inline auto p_b3CalculateMassMatrixCommandInit=reinterpret_cast<decltype(&::b3CalculateMassMatrixCommandInit)>(dlsym(engine,"b3CalculateMassMatrixCommandInit"));
inline auto p_b3CalculateMassMatrixSetFlags=reinterpret_cast<decltype(&::b3CalculateMassMatrixSetFlags)>(dlsym(engine,"b3CalculateMassMatrixSetFlags"));
inline auto p_b3ChangeDynamicsInfoSetAngularDamping=reinterpret_cast<decltype(&::b3ChangeDynamicsInfoSetAngularDamping)>(dlsym(engine,"b3ChangeDynamicsInfoSetAngularDamping"));
inline auto p_b3ChangeDynamicsInfoSetAnisotropicFriction=reinterpret_cast<decltype(&::b3ChangeDynamicsInfoSetAnisotropicFriction)>(dlsym(engine,"b3ChangeDynamicsInfoSetAnisotropicFriction"));
inline auto p_b3ChangeDynamicsInfoSetContactStiffnessAndDamping=reinterpret_cast<decltype(&::b3ChangeDynamicsInfoSetContactStiffnessAndDamping)>(dlsym(engine,"b3ChangeDynamicsInfoSetContactStiffnessAndDamping"));
inline auto p_b3ChangeDynamicsInfoSetJointDamping=reinterpret_cast<decltype(&::b3ChangeDynamicsInfoSetJointDamping)>(dlsym(engine,"b3ChangeDynamicsInfoSetJointDamping"));
inline auto p_b3ChangeDynamicsInfoSetLateralFriction=reinterpret_cast<decltype(&::b3ChangeDynamicsInfoSetLateralFriction)>(dlsym(engine,"b3ChangeDynamicsInfoSetLateralFriction"));
inline auto p_b3ChangeDynamicsInfoSetLinearDamping=reinterpret_cast<decltype(&::b3ChangeDynamicsInfoSetLinearDamping)>(dlsym(engine,"b3ChangeDynamicsInfoSetLinearDamping"));
inline auto p_b3ConnectPhysicsDirect=reinterpret_cast<decltype(&::b3ConnectPhysicsDirect)>(dlsym(engine,"b3ConnectPhysicsDirect"));
inline auto p_b3CreatePoseCommandInit=reinterpret_cast<decltype(&::b3CreatePoseCommandInit)>(dlsym(engine,"b3CreatePoseCommandInit"));
inline auto p_b3CreatePoseCommandSetBaseAngularVelocity=reinterpret_cast<decltype(&::b3CreatePoseCommandSetBaseAngularVelocity)>(dlsym(engine,"b3CreatePoseCommandSetBaseAngularVelocity"));
inline auto p_b3CreatePoseCommandSetBaseLinearVelocity=reinterpret_cast<decltype(&::b3CreatePoseCommandSetBaseLinearVelocity)>(dlsym(engine,"b3CreatePoseCommandSetBaseLinearVelocity"));
inline auto p_b3CreatePoseCommandSetBaseOrientation=reinterpret_cast<decltype(&::b3CreatePoseCommandSetBaseOrientation)>(dlsym(engine,"b3CreatePoseCommandSetBaseOrientation"));
inline auto p_b3CreatePoseCommandSetBasePosition=reinterpret_cast<decltype(&::b3CreatePoseCommandSetBasePosition)>(dlsym(engine,"b3CreatePoseCommandSetBasePosition"));
inline auto p_b3CreatePoseCommandSetJointPosition=reinterpret_cast<decltype(&::b3CreatePoseCommandSetJointPosition)>(dlsym(engine,"b3CreatePoseCommandSetJointPosition"));
inline auto p_b3CreatePoseCommandSetJointVelocity=reinterpret_cast<decltype(&::b3CreatePoseCommandSetJointVelocity)>(dlsym(engine,"b3CreatePoseCommandSetJointVelocity"));
inline auto p_b3DisconnectSharedMemory=reinterpret_cast<decltype(&::b3DisconnectSharedMemory)>(dlsym(engine,"b3DisconnectSharedMemory"));
inline auto p_b3GetJointInfo=reinterpret_cast<decltype(&::b3GetJointInfo)>(dlsym(engine,"b3GetJointInfo"));
inline auto p_b3GetNumJoints=reinterpret_cast<decltype(&::b3GetNumJoints)>(dlsym(engine,"b3GetNumJoints"));
inline auto p_b3GetStatusActualState=reinterpret_cast<decltype(&::b3GetStatusActualState)>(dlsym(engine,"b3GetStatusActualState"));
inline auto p_b3GetStatusBodyIndex=reinterpret_cast<decltype(&::b3GetStatusBodyIndex)>(dlsym(engine,"b3GetStatusBodyIndex"));
inline auto p_b3GetStatusGetStateId=reinterpret_cast<decltype(&::b3GetStatusGetStateId)>(dlsym(engine,"b3GetStatusGetStateId"));
inline auto p_b3GetStatusMassMatrix=reinterpret_cast<decltype(&::b3GetStatusMassMatrix)>(dlsym(engine,"b3GetStatusMassMatrix"));
inline auto p_b3InitChangeDynamicsInfo=reinterpret_cast<decltype(&::b3InitChangeDynamicsInfo)>(dlsym(engine,"b3InitChangeDynamicsInfo"));
inline auto p_b3InitPhysicsParamCommand=reinterpret_cast<decltype(&::b3InitPhysicsParamCommand)>(dlsym(engine,"b3InitPhysicsParamCommand"));
inline auto p_b3InitStepSimulationCommand=reinterpret_cast<decltype(&::b3InitStepSimulationCommand)>(dlsym(engine,"b3InitStepSimulationCommand"));
inline auto p_b3InvertTransform=reinterpret_cast<decltype(&::b3InvertTransform)>(dlsym(engine,"b3InvertTransform"));
inline auto p_b3JointControlCommandInit2=reinterpret_cast<decltype(&::b3JointControlCommandInit2)>(dlsym(engine,"b3JointControlCommandInit2"));
inline auto p_b3JointControlSetDesiredForceTorque=reinterpret_cast<decltype(&::b3JointControlSetDesiredForceTorque)>(dlsym(engine,"b3JointControlSetDesiredForceTorque"));
inline auto p_b3JointControlSetDesiredPosition=reinterpret_cast<decltype(&::b3JointControlSetDesiredPosition)>(dlsym(engine,"b3JointControlSetDesiredPosition"));
inline auto p_b3JointControlSetDesiredVelocity=reinterpret_cast<decltype(&::b3JointControlSetDesiredVelocity)>(dlsym(engine,"b3JointControlSetDesiredVelocity"));
inline auto p_b3JointControlSetKd=reinterpret_cast<decltype(&::b3JointControlSetKd)>(dlsym(engine,"b3JointControlSetKd"));
inline auto p_b3JointControlSetKp=reinterpret_cast<decltype(&::b3JointControlSetKp)>(dlsym(engine,"b3JointControlSetKp"));
inline auto p_b3JointControlSetMaximumForce=reinterpret_cast<decltype(&::b3JointControlSetMaximumForce)>(dlsym(engine,"b3JointControlSetMaximumForce"));
inline auto p_b3LoadStateCommandInit=reinterpret_cast<decltype(&::b3LoadStateCommandInit)>(dlsym(engine,"b3LoadStateCommandInit"));
inline auto p_b3LoadStateSetStateId=reinterpret_cast<decltype(&::b3LoadStateSetStateId)>(dlsym(engine,"b3LoadStateSetStateId"));
inline auto p_b3LoadUrdfCommandInit=reinterpret_cast<decltype(&::b3LoadUrdfCommandInit)>(dlsym(engine,"b3LoadUrdfCommandInit"));
inline auto p_b3LoadUrdfCommandSetFlags=reinterpret_cast<decltype(&::b3LoadUrdfCommandSetFlags)>(dlsym(engine,"b3LoadUrdfCommandSetFlags"));
inline auto p_b3MultiplyTransforms=reinterpret_cast<decltype(&::b3MultiplyTransforms)>(dlsym(engine,"b3MultiplyTransforms"));
inline auto p_b3PhysicsParamSetEnableConeFriction=reinterpret_cast<decltype(&::b3PhysicsParamSetEnableConeFriction)>(dlsym(engine,"b3PhysicsParamSetEnableConeFriction"));
inline auto p_b3PhysicsParamSetGravity=reinterpret_cast<decltype(&::b3PhysicsParamSetGravity)>(dlsym(engine,"b3PhysicsParamSetGravity"));
inline auto p_b3PhysicsParamSetNumSolverIterations=reinterpret_cast<decltype(&::b3PhysicsParamSetNumSolverIterations)>(dlsym(engine,"b3PhysicsParamSetNumSolverIterations"));
inline auto p_b3PhysicsParamSetTimeStep=reinterpret_cast<decltype(&::b3PhysicsParamSetTimeStep)>(dlsym(engine,"b3PhysicsParamSetTimeStep"));
inline auto p_b3RequestActualStateCommandInit=reinterpret_cast<decltype(&::b3RequestActualStateCommandInit)>(dlsym(engine,"b3RequestActualStateCommandInit"));
inline auto p_b3SaveStateCommandInit=reinterpret_cast<decltype(&::b3SaveStateCommandInit)>(dlsym(engine,"b3SaveStateCommandInit"));
inline auto p_b3SubmitClientCommandAndWaitStatus=reinterpret_cast<decltype(&::b3SubmitClientCommandAndWaitStatus)>(dlsym(engine,"b3SubmitClientCommandAndWaitStatus"));
}
#define b3CollisionFilterCommandInit native_reference::p_b3CollisionFilterCommandInit
#define b3SetCollisionFilterGroupMask native_reference::p_b3SetCollisionFilterGroupMask
#define b3CalculateMassMatrixCommandInit native_reference::p_b3CalculateMassMatrixCommandInit
#define b3CalculateMassMatrixSetFlags native_reference::p_b3CalculateMassMatrixSetFlags
#define b3ChangeDynamicsInfoSetAngularDamping native_reference::p_b3ChangeDynamicsInfoSetAngularDamping
#define b3ChangeDynamicsInfoSetAnisotropicFriction native_reference::p_b3ChangeDynamicsInfoSetAnisotropicFriction
#define b3ChangeDynamicsInfoSetContactStiffnessAndDamping native_reference::p_b3ChangeDynamicsInfoSetContactStiffnessAndDamping
#define b3ChangeDynamicsInfoSetJointDamping native_reference::p_b3ChangeDynamicsInfoSetJointDamping
#define b3ChangeDynamicsInfoSetLateralFriction native_reference::p_b3ChangeDynamicsInfoSetLateralFriction
#define b3ChangeDynamicsInfoSetLinearDamping native_reference::p_b3ChangeDynamicsInfoSetLinearDamping
#define b3ConnectPhysicsDirect native_reference::p_b3ConnectPhysicsDirect
#define b3CreatePoseCommandInit native_reference::p_b3CreatePoseCommandInit
#define b3CreatePoseCommandSetBaseAngularVelocity native_reference::p_b3CreatePoseCommandSetBaseAngularVelocity
#define b3CreatePoseCommandSetBaseLinearVelocity native_reference::p_b3CreatePoseCommandSetBaseLinearVelocity
#define b3CreatePoseCommandSetBaseOrientation native_reference::p_b3CreatePoseCommandSetBaseOrientation
#define b3CreatePoseCommandSetBasePosition native_reference::p_b3CreatePoseCommandSetBasePosition
#define b3CreatePoseCommandSetJointPosition native_reference::p_b3CreatePoseCommandSetJointPosition
#define b3CreatePoseCommandSetJointVelocity native_reference::p_b3CreatePoseCommandSetJointVelocity
#define b3DisconnectSharedMemory native_reference::p_b3DisconnectSharedMemory
#define b3GetJointInfo native_reference::p_b3GetJointInfo
#define b3GetNumJoints native_reference::p_b3GetNumJoints
#define b3GetStatusActualState native_reference::p_b3GetStatusActualState
#define b3GetStatusBodyIndex native_reference::p_b3GetStatusBodyIndex
#define b3GetStatusGetStateId native_reference::p_b3GetStatusGetStateId
#define b3GetStatusMassMatrix native_reference::p_b3GetStatusMassMatrix
#define b3InitChangeDynamicsInfo native_reference::p_b3InitChangeDynamicsInfo
#define b3InitPhysicsParamCommand native_reference::p_b3InitPhysicsParamCommand
#define b3InitStepSimulationCommand native_reference::p_b3InitStepSimulationCommand
#define b3InvertTransform native_reference::p_b3InvertTransform
#define b3JointControlCommandInit2 native_reference::p_b3JointControlCommandInit2
#define b3JointControlSetDesiredForceTorque native_reference::p_b3JointControlSetDesiredForceTorque
#define b3JointControlSetDesiredPosition native_reference::p_b3JointControlSetDesiredPosition
#define b3JointControlSetDesiredVelocity native_reference::p_b3JointControlSetDesiredVelocity
#define b3JointControlSetKd native_reference::p_b3JointControlSetKd
#define b3JointControlSetKp native_reference::p_b3JointControlSetKp
#define b3JointControlSetMaximumForce native_reference::p_b3JointControlSetMaximumForce
#define b3LoadStateCommandInit native_reference::p_b3LoadStateCommandInit
#define b3LoadStateSetStateId native_reference::p_b3LoadStateSetStateId
#define b3LoadUrdfCommandInit native_reference::p_b3LoadUrdfCommandInit
#define b3LoadUrdfCommandSetFlags native_reference::p_b3LoadUrdfCommandSetFlags
#define b3MultiplyTransforms native_reference::p_b3MultiplyTransforms
#define b3PhysicsParamSetEnableConeFriction native_reference::p_b3PhysicsParamSetEnableConeFriction
#define b3PhysicsParamSetGravity native_reference::p_b3PhysicsParamSetGravity
#define b3PhysicsParamSetNumSolverIterations native_reference::p_b3PhysicsParamSetNumSolverIterations
#define b3PhysicsParamSetTimeStep native_reference::p_b3PhysicsParamSetTimeStep
#define b3RequestActualStateCommandInit native_reference::p_b3RequestActualStateCommandInit
#define b3SaveStateCommandInit native_reference::p_b3SaveStateCommandInit
#define b3SubmitClientCommandAndWaitStatus native_reference::p_b3SubmitClientCommandAndWaitStatus
#endif
