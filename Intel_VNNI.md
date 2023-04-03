 #인텔의 머신러닝용 명령어

인텔은 2세대 제온 SP(캐스케이드 레이크)에서 머신러닝 추론 처리에 효과적인 명령어군 DL Boost를 지원합니다,
그리고 2020년 출시 예정인 차세대 Xeon SP(Cooper Lake)에서 bfloat16 명령어를 지원합니다.
여기서는 Xbyak을 이용한 이들 명령어의 활용 방법을 소개합니다.

## 샘플 코드
- [vpdpbusd-test.cpp](https://github.com/herumi/misc/blob/master/avx-512/vpdpbusd-test.cpp)
  - Cascade Lake용 8bit 적산 연산 명령어 `vpdpbusd` 샘플
- [bfloat16-test.cpp](https://github.com/herumi/misc/blob/master/avx-512/bfloat16-test.cpp)
  - Cooper Lake용 bfloat16용 적분 연산 명령어 `vdpbf16ps`와 변환 명령어 `vcvtne2ps2bf16` 샘플

## 에뮬레이터
Cooper Lake는 아직 출시되지 않았기 때문에 에뮬레이터를 사용하여 동작을 확인합니다.
[Intel Software Development Emulator Download](https://software.intel.com/content/www/us/en/develop/articles/pre-release-license- agreement-for-intel-software-development-emulator-accept-end-user-license-agreement-and-download.html)에서
Intel SDE를 다운로드합니다.

`````.
sde -cpx -- 샘플 프로그램
````을 실행합니다.
로 실행합니다. cpx`는 Cooper Lake 에뮬레이션을 의미합니다.

## vpdpbusd
vpdpbusd는 Cascade Lake에서 지원하는 8bit 정수 간 적분 연산 명령어이다.

좀 더 정확히 말하면 8bit 부호 없는 정수와 8bit 부호 있는 정수 4개 간의 적산을 구한다,
그 결과(int로 확장됩니다)를 목적지 레지스터에 가산합니다.
AVX-512는 8bit가 64개이므로 이 연산을 16번 병렬로 실행합니다.

C로 작성하면 다음과 같은 동작을 합니다.


void vpdpbusdC(int *dst, const uint8_t *u, const int8_t *s)
{
    for (int i = 0; i < 16; i++) {
        int sum = dst[i];
        for (int j = 0; j < 4; j++) {
            sum += u[i * 4 + j] * s[i * 4 + j]; }
        }
        dst[i] = sum; }
    }
}
```

```
dst[ 0] += u[ 0] * s[ 0] * s[ 0] + u[ 1] * s[ 1] + u[ 2] * s[ 2] + u[ 3] * s[ 3];
dst[ 1] += u[ 4] * s[ 4] + u[ 5] * s[ 5] + u[ 6] * s[ 6] * s[ 6] + u[ 7] * s[ 7];
dst[ 2] += u[ 8] * s[ 8] * s[ 8] + u[ 9] * s[ 9] + u[10] * s[10] + u[10] * s[11] * s[11];
...
dst[15] += u[60] * s[60] + u[61] * s[61] + u[61] * s[62] * s[62] + u[63] * s[63]; ...
```

## bfloat16
bfloat16은 기계학습의 계산에서 기존의 float(32bit)만큼의 정밀도가 필요하지 않은 곳에서 활용되는 16bit 부동소수점수입니다.

float가 1bit의 부호, 8bit의 지수부, 23bit의 가수부인 반면 bfloat16은 부호와 지수부가 동일하고 가수부를 상위 7bit로 줄인 포맷으로 표현됩니다.

float와 bfloat16의 포맷표
타입| 부호 비트(s) | 지수부(e) | 가숫대(f) | 값
-|-|-|-|-|-|-|-|
float|1|8|23|(-1)^s 2^(e-127)×(1 + f/2^24)|
bfloat16|1|8|7|(-1)^s 2^(e-127)×(1 + f/2^8)|

### float와 bfloat16의 상호 변환

준비로 float와 uint32_t의 상호 변환 함수를 준비한다.

union fi {
    float f;
    uint32_t u;
};

typedef uint16_t bfloat16;
loat u2f(uint32_t u)
{
    fi fi;
    fi.u = u;
    return fi.f;
}

uint32_t f2u(float f)
{
    fi fi;
    fi.f = f;
    return fi.u;
}
````.

bfloat16에서 float로 변환하려면 16bit 정수로 좌변 16bit 시프트하면 된다.


float bfloat16_to_float(bfloat16 f)
{ f
    return u2f(f << 16);
}
```

float에서 bfloat16도 가수의 하위 16bit를 잘라내기만 해도 되지만, 반올림 처리를 하면 조금 더 정확도가 높아진다.


bfloat16 float_to_bfloat16(float f)
{
    // ignore denormal and infinity
    uint32_t u = f2u(f);
    uint32_t rounding = 0x7fff + ((u >> 16) & 1);
    u += rounding;
    return bfloat16(u >> 16);
}
```
여기서는 denormal이나 infinity 상태는 무시했습니다. 엄격하게 하려면 [tensorflow의 bfloat16.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/bfloat16/bfloat16.h)를 참고하시기 바랍니다. 하시기 바랍니다.

### `vcvtne2ps2bf16`

````
vcvtne2ps2bf16 dst, src1, src2
```` ```
은 src1, src2에서 지정한 float 타입의 SIMD 레지스터를 bfloat16 타입으로 변환하여 dst에 대입합니다.
C에서의 코드는 다음과 같다.


void vcvtne2ps2bf16C(bfloat16 *dst, const float *src1, const float *src2)
{
    for (int i = 0; i < 16; i++) {
        dst[i] = float_to_bfloat16(src1[i]);
        dst[i+16] = float_to_bfloat16(src2[i]);
    }
}
```

### `vdpbf16ps`
vdpbf16ps는 bfloat16 타입의 SIMD 레지스터 2개를 가져와 float로 변환한 후 그 합계를 목적지 레지스터에 더한다.

C로 작성된 코드는 다음과 같다.


void vdpbf16psC(float *dst, const bfloat16 *src1, const bfloat16 *src2)
{
    for (int i = 0; i < 16; i++) {
        dst[i] += bfloat16_to_float(src1[i*2+0]) * bfloat16_to_float(src2[i*2+0]);
        dst[i] += bfloat16_to_float(src1[i*2+1]) * bfloat16_to_float(src2[i*2+1]);
    }
}
```

```

dst[ 1] = float(src1[ 2]) * float(src2[ 2]) + float(src1[ 3]) * float(src2[ 3]); dst[ 1] = float(src1[ 2]) * float(src2[ 2])
dst[ 2] = float(src1[ 4]) * float(src2[ 4]) + float(src1[ 5]) * float(src2[ 5]); ...
...
dst[15] = float(src1[30]) * float(src2[30]) + float(src1[31]) * float(src2[31]); ...
````

## CPU 판별
vpdpbusd`는 Cascade Lake 이상, `vdpbf16ps` 등의 명령어는 Cooper Lake 이상이어야 실행할 수 있다.
Xbyak은 CPU 판별을 위한 클래스 `Xbyak::util::CPU`를 가지고 있다.

CPU의 이용 플래그 표
명령어|이용 플래그|대응 CPU|
-|-|-|-|
`vpdpbusd`|`Xbyak::util::Cpu::tAVX512_VNNI`|Cascade Lake
`vdpbf16ps`|`Xbyak::util::Cpu::tAVX512_BF16`|Cooper Lake

구체적으로 다음 코드에서 대응할 수 없는 오류로 만들거나 대체 코드를 작성합니다.


    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAVX512_VNNI)) {
        printf("AVX512_VNNI is not supported\n");
        return false;
    }
    if (!cpu.has(Xbyak::util::Cpu::Cpu::tAVX512_BF16)) {
        printf("AVX512_BF16 is not supported\n");
        return false;
    }
```
