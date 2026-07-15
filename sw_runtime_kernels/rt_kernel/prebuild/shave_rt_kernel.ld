OUTPUT_FORMAT("elf32-sparc")
OUTPUT_ARCH(sparc)

/* Default window base address; can be overridden at link time with -defsym. */
shave_win_a = DEFINED(shave_win_a) ? shave_win_a : 0x1C000000;

MEMORY {
    WIN_A (rx) : ORIGIN = shave_win_a, LENGTH = 0x1000000
}

SECTIONS
{
    .text : {
        *(.text*)
        *(S.text.*)
        . = ALIGN(16);
        *(.critical*)
        . = ALIGN(16);
        *(.preemption*)
        /* Covers all linkonce groups (text, rodata, data) emitted by older toolchains. */
        *(.gnu.linkonce.*)
        . = ALIGN(16);
        *(.rodata*)
        KEEP(*(.uuid.rodata*))
    } > WIN_A

    .versiondata : {
        KEEP(*(.versiondata))
    } > WIN_A

    /DISCARD/ : {
        *(.data*)
        *(.bss*)
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
