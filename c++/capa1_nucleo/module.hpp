#ifndef CAPA1_NUCLEO_MODULE_HPP
#define CAPA1_NUCLEO_MODULE_HPP


#include "tensor.hpp"

// ------------------------------------------------------------
// TODA clase de la red hereda de Module e implementa forward().
// Regla: nunca modificar esta interfaz sin avisar a todo el equipo.
// ------------------------------------------------------------

class Module {
public:
    // Cada subclase implementa su propia lógica de forward.
    virtual Tensor forward(const Tensor& x) = 0;

    // Destructor virtual para que delete funcione correctamente
    virtual ~Module() = default;
};

#endif  // CAPA1_NUCLEO_MODULE_HPP
