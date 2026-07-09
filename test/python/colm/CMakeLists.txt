include(ParflowTest)

set(COLM_TESTS)

if(${PARFLOW_HAVE_CLM})
    if(${PARFLOW_HAVE_HYPRE})
        list(APPEND COLM_TESTS colm_ci)
    endif()
endif()

foreach(inputfile ${COLM_TESTS})
    pf_add_py_test(${inputfile})
endforeach()
