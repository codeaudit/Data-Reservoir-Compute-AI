#include "jni.bi"

sub hsixteen naked (x as single ptr, n as ulongint,scale as single)
asm	
	shufps xmm0,xmm0,0
	.align 16
h16:
	subq rsi,16
	movups xmm1,[rdi]
	movups xmm2,[rdi+16]
	movups xmm3,[rdi+2*16]
	movups xmm4,[rdi+3*16]
	movups xmm5,xmm1
	movups xmm6,xmm3
	haddps xmm1,xmm2
	haddps xmm3,xmm4
	hsubps xmm5,xmm2
	hsubps xmm6,xmm4
	movups xmm2,xmm1
	movups xmm4,xmm3
	haddps xmm1,xmm5
	haddps xmm3,xmm6
	hsubps xmm2,xmm5
	hsubps xmm4,xmm6
	movups xmm5,xmm1
	movups xmm6,xmm3
	haddps xmm1,xmm2
	haddps xmm3,xmm4
	hsubps xmm5,xmm2
	hsubps xmm6,xmm4
	movups xmm2,xmm1
	movups xmm4,xmm5
	addps xmm1,xmm3
	addps xmm5,xmm6
	subps xmm2,xmm3
	subps xmm4,xmm6
	mulps xmm1,xmm0
	mulps xmm5,xmm0
	mulps xmm2,xmm0
	mulps xmm4,xmm0
	movups [rdi],xmm1
	movups [rdi+16],xmm5
	movups [rdi+2*16],xmm2
	movups [rdi+3*16],xmm4
	lea rdi,[rdi+64]
	jnz h16
	ret
end asm
end sub

sub hgap naked (x as single ptr,gap as ulongint,n as ulongint)
asm
    movq rcx,rsi
	lea r8,[rdi+4*rsi]
	shr rdx,1
	.align 16	
hgaploop:
	subq rcx,16
	movups xmm0,[rdi]
	movups xmm1,[rdi+16]
	movups xmm2,[rdi+2*16]
	movups xmm3,[rdi+3*16]
	movups xmm8,[r8]
	movups xmm9,[r8+16]
	movups xmm10,[r8+2*16]
	movups xmm11,[r8+3*16]
	movups xmm4,xmm0
	movups xmm5,xmm1
	movups xmm6,xmm2
	movups xmm7,xmm3
	addps xmm0,xmm8
	addps xmm1,xmm9
	addps xmm2,xmm10
	addps xmm3,xmm11
	subps xmm4,xmm8
	subps xmm5,xmm9
	subps xmm6,xmm10
	subps xmm7,xmm11
	movups [rdi],xmm0
	movups [rdi+16],xmm1
	movups [rdi+2*16],xmm2
	movups [rdi+3*16],xmm3
	movups [r8],xmm4
	movups [r8+16],xmm5
	movups [r8+2*16],xmm6
	movups [r8+3*16],xmm7
	lea rdi,[rdi+64]
	lea r8,[r8+64]
	jnz hgaploop
	subq rdx,rsi
	movq rcx,rsi
	movq rdi,r8
	lea r8,[r8+4*rsi]
	jnz hgaploop
	ret
end asm
end sub

' n must be a power of 2, 16 or over 16,32,64....
sub wht(vec as single ptr, n as ulongint)
	   const lim as ulongint=8192
	   dim as ulongint gap,k
	   dim as single scale=1.0/sqr(n)
	   k=n
	   if k>lim then k=lim
	   for i as ulongint=0 to n-1 step lim
		   hsixteen(vec+i,k,scale)
		   gap=16
		   while gap<k
			  hgap(vec+i,gap,k)
			  gap+=gap
		   wend
		next
		while gap<n
			hgap(vec,gap,n)
			gap+=gap
		wend	
end sub

sub signFlip naked (result as single ptr,x as single ptr,h as ulongint,n as ulongint)
asm
	movq rax,rndphi[rip]
	movq r8,rndsqr3[rip]
	imulq rdx,rax
	movdqu xmm8,flipshift[rip]
	movdqu xmm9,flipshift[rip+16]
	add rdx,r8
	movdqu xmm10,flipshift[rip+32]
	imulq rdx,rax
	movdqu xmm11,flipshift[rip+48]
	movdqu xmm12,flipmask[rip]
	movd xmm4,edx
	.align 16
flipAlp:
	imulq rdx,rax
	pshufd xmm4,xmm4,0
	movdqu xmm0,[rsi]
	movdqu xmm1,[rsi+16]
	movdqu xmm2,[rsi+2*16]
	movdqu xmm3,[rsi+3*16]
	addq rdx,r8
	movdqa xmm5,xmm4
	movdqa xmm6,xmm4
	movdqa xmm7,xmm4
	imulq rdx,rax
	pmulld xmm4,xmm8
	pmulld xmm5,xmm9
	pmulld xmm6,xmm10
	pmulld xmm7,xmm11
	addq rdx,r8
	pand xmm4,xmm12
	pand xmm5,xmm12
	pand xmm6,xmm12
	pand xmm7,xmm12
	bswapq rdx
	pxor xmm0,xmm4
	pxor xmm1,xmm5
	pxor xmm2,xmm6
	pxor xmm3,xmm7
	movd xmm4,edx
	sub rcx,16
	movdqu [rdi],xmm0
	movdqu [rdi+16],xmm1
	movdqu [rdi+2*16],xmm2
	movdqu [rdi+3*16],xmm3
	lea rsi,[rsi+64]
	lea rdi,[rdi+64]
	jnz flipAlp
	ret
 flipshift:   .int 1,2,4,8,16,32,64,128
					.int 256,512,1024,2048,4096,8192,16384,32768
 flipmask:	.int 0x80000000,0x80000000,0x80000000,0x80000000
 rndphi:	   	.quad 0x9E3779B97F4A7C15
 rndsqr3:	.quad 0xBB67AE8584CAA73B
end asm
end sub

'less than zero=-1, positive =1
sub signOf naked (result as single ptr,x as single ptr, n as ulongint)
asm
	mov eax,0x3f800000
	mov ecx,0x80000000
	movd xmm4,eax
	movd xmm5,ecx
	shufps xmm4,xmm4,0
	shufps xmm5,xmm5,0
	.align 16
signoflp:
	movups xmm0,[rsi]
	movups xmm1,[rsi+16]
	movups xmm2,[rsi+16*2]
	movups xmm3,[rsi+16*3]
	subq rdx,16
	andps xmm0,xmm5
	andps xmm1,xmm5
	andps xmm2,xmm5
	andps xmm3,xmm5
	por xmm0,xmm4
	por xmm1,xmm4
	por xmm2,xmm4
	por xmm3,xmm4
	movups [rdi],xmm0
	movups [rdi+16],xmm1
	movups [rdi+16*2],xmm2
	movups [rdi+16*3],xmm3
	lea rsi,[rsi+64]
	lea rdi,[rdi+64]
	jnz signoflp
	ret
end asm
end sub

sub Java_s6regen_WHT_whtNative alias "Java_data_reservoir_compute_ai_WHT_whtNative" (env as JNIEnv ptr,cl as jclass,array as jarray)
dim as long n=(*env)->GetArrayLength(env,array)
dim as single ptr vec=(*env)->GetPrimitiveArrayCritical (env,array,0)
wht(vec,n)
(*env)->ReleasePrimitiveArrayCritical(env,array,vec,0)
end sub

sub Java_s6regen_WHT_signOfNative alias "Java_data_reservoir_compute_ai_WHT_signOfNative" (env as JNIEnv ptr,cl as jclass,array1 as jarray,array2 as jarray)
dim as long n=(*env)->GetArrayLength(env,array1)
dim as single ptr vec1=(*env)->GetPrimitiveArrayCritical (env,array1,0)
dim as single ptr vec2=(*env)->GetPrimitiveArrayCritical (env,array2,0)
signOf(vec1,vec2,n)
(*env)->ReleasePrimitiveArrayCritical(env,array2,vec2,0)
(*env)->ReleasePrimitiveArrayCritical(env,array1,vec1,0)
end sub

sub Java_s6regen_WHT_signFlipNative alias "Java_data_reservoir_compute_ai_WHT_signFlipNative" (env as JNIEnv ptr,cl as jclass,array as jarray,h as jlong)
dim as long n=(*env)->GetArrayLength(env,array)
dim as single ptr vec=(*env)->GetPrimitiveArrayCritical (env,array,0)
signFlip(vec,vec,h,n)
(*env)->ReleasePrimitiveArrayCritical(env,array,vec,0)
end sub

