#pragma once

#include "tensor.cuh"

// ------------------------------------------------------------
// TODA clase de la red hereda de Module e implementa forward().
// Regla: nunca modificar esta interfaz sin avisar a todo el equipo.
// ------------------------------------------------------------

class Module {
public:
    // Cada subclase implementa su propia logica de forward.
    virtual Tensor forward(const Tensor& x) = 0;

    // Destructor virtual para que delete funcione correctamente
    virtual ~Module() = default;
};
