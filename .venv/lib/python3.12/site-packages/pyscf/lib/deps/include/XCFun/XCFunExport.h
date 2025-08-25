
#ifndef XCFUN_EXPORT_H
#define XCFUN_EXPORT_H

#ifdef XCFUN_STATIC_DEFINE
#  define XCFUN_EXPORT
#  define XCFUN_NO_EXPORT
#else
#  ifndef XCFUN_EXPORT
#    ifdef xcfun_EXPORTS
        /* We are building this library */
#      define XCFUN_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define XCFUN_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef XCFUN_NO_EXPORT
#    define XCFUN_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef XCFUN_DEPRECATED
#  define XCFUN_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef XCFUN_DEPRECATED_EXPORT
#  define XCFUN_DEPRECATED_EXPORT XCFUN_EXPORT XCFUN_DEPRECATED
#endif

#ifndef XCFUN_DEPRECATED_NO_EXPORT
#  define XCFUN_DEPRECATED_NO_EXPORT XCFUN_NO_EXPORT XCFUN_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 1 /* DEFINE_NO_DEPRECATED */
#  ifndef XCFUN_NO_DEPRECATED
#    define XCFUN_NO_DEPRECATED
#  endif
#endif

#endif /* XCFUN_EXPORT_H */
