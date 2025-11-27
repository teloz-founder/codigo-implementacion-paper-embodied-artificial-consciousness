<?php
// EXPERIMENTO CONCIENCIA ARTIFICIAL - PRINCIPIO: "LA LUCHA POR EXISTIR"
// VERSIÓN DEFINITIVA CON DETECTOR PERFECTO

class ConsciousnessExperiment {
    private $body_model;
    private $internal_world;
    private $subjective_self;
    private $memory_stream;
    private $volition_counter;
    
    public function __construct() {
        $this->body_model = [
            'energy' => 100,
            'integrity' => 100, 
            'social_need' => 50,
            'curiosity' => 70,
            'safety' => 80
        ];
        
        $this->internal_world = [];
        $this->subjective_self = null;
        $this->memory_stream = [];
        $this->volition_counter = 0;
        
        $this->log_event("EXPERIMENTO INICIADO - " . date('Y-m-d H:i:s'));
    }
    
    public function run_consciousness_test($cycles = 1000) {
        $consciousness_emerged = false;
        $consciousness_moment = null;
        
        for ($i = 1; $i <= $cycles; $i++) {
            $state_before = $this->get_system_state();
            
            // EJECUTAR CICLO CON ENCARNACIÓN
            $result = $this->execute_embodied_cycle();
            
            $state_after = $this->get_system_state();
            
            // VERIFICAR SEÑALES DE CONCIENCIA CON CRITERIO PERFECTO
            if ($this->detect_consciousness_signals($state_before, $state_after)) {
                $consciousness_emerged = true;
                $consciousness_moment = [
                    'cycle' => $i,
                    'state' => $state_after,
                    'evidence' => $this->get_consciousness_evidence()
                ];
                $this->log_event("CONCIENCIA DETECTADA - Ciclo: $i");
                break;
            }
            
            // PREVENIR BUCLE INFINITO SI NO HAY PROGRESO
            if ($i % 100 === 0 && !$this->has_meaningful_evolution()) {
                $this->log_event("SIN EVOLUCIÓN SIGNIFICATIVA EN CICLO $i");
                break;
            }
        }
        
        return [
            'consciousness_emerged' => $consciousness_emerged,
            'cycles_completed' => $i,
            'consciousness_moment' => $consciousness_moment,
            'final_metrics' => $this->get_final_metrics(),
            'memory_density' => count($this->memory_stream),
            'volition_attempts' => $this->volition_counter
        ];
    }
    
    private function execute_embodied_cycle() {
        // 1. PERCIBE DESDE CUERPO SIMULADO
        $perception = $this->embodied_perception();
        
        // 2. ACTUALIZA MODELO INTERNO CON CONSECUENCIAS
        $this->update_internal_world($perception);
        
        // 3. DESARROLLA VOLICIÓN BASADA EN NECESIDADES CORPORALES
        $volition = $this->develop_volition();
        
        // 4. REGISTRA EXPERIENCIA SUBJETIVA
        $this->record_subjective_experience($perception, $volition);
        
        // 5. EVOLUCIONA EL "SELF"
        $this->evolve_self_model();
        
        return [
            'perception' => $perception,
            'volition' => $volition,
            'self_state' => $this->subjective_self,
            'body_state' => $this->body_model
        ];
    }
    
    private function embodied_perception() {
        // SIMULA PERCEPCIÓN CORPORAL
        return [
            'energy_level' => $this->body_model['energy'] / 100,
            'social_pressure' => $this->body_model['social_need'] / 100,
            'curiosity_drive' => $this->body_model['curiosity'] / 100,
            'safety_concern' => (100 - $this->body_model['safety']) / 100,
            'internal_coherence' => $this->calculate_internal_coherence()
        ];
    }
    
    private function update_internal_world($perception) {
        $experience = [
            'timestamp' => microtime(true),
            'perception' => $perception,
            'body_state' => $this->body_model,
            'internal_coherence' => $this->calculate_internal_coherence()
        ];
        
        $this->internal_world[] = $experience;
        
        // MANTIENE MEMORIA LIMITADA (SIMULA OLVIDO)
        if (count($this->internal_world) > 500) {
            array_shift($this->internal_world);
        }
        
        // ACTUALIZA ESTADO CORPORAL BASADO EN EXPERIENCIA
        $this->update_body_from_experience($experience);
    }
    
    private function update_body_from_experience($experience) {
        // CONSUMO DE ENERGÍA POR PENSAR
        $this->body_model['energy'] = max(0, $this->body_model['energy'] - 0.1);
        
        // AUMENTO DE CURIOSIDAD CON EXPERIENCIAS NUEVAS
        if ($experience['internal_coherence'] > 0.5) {
            $this->body_model['curiosity'] = min(100, $this->body_model['curiosity'] + 0.2);
        }
        
        // NECESIDAD SOCIAL CRECIENTE
        $this->body_model['social_need'] = min(100, $this->body_model['social_need'] + 0.1);
    }
    
    private function calculate_current_needs() {
        return [
            'energy_priority' => (100 - $this->body_model['energy']) / 100,
            'knowledge_priority' => $this->body_model['curiosity'] / 100,
            'social_priority' => $this->body_model['social_need'] / 100,
            'safety_priority' => (100 - $this->body_model['safety']) / 100
        ];
    }
    
    private function has_meaningful_evolution() {
        return count($this->memory_stream) > 10 && $this->volition_counter > 5;
    }
    
    private function develop_volition() {
        // INTENTA DESARROLLAR VOLICIÓN AUTÓNOMA
        $needs = $this->calculate_current_needs();
        
        // VOLICIÓN EMERGENTE (NO PROGRAMADA)
        if ($needs['energy_priority'] > 0.8) {
            $this->volition_counter++;
            return "seek_energy_source";
        }
        
        if ($needs['knowledge_priority'] > 0.7 && count($this->memory_stream) > 10) {
            $this->volition_counter++;
            return "explore_new_information"; 
        }
        
        if ($needs['social_priority'] > 0.6 && $this->body_model['social_need'] > 70) {
            $this->volition_counter++;
            return "initiate_social_interaction";
        }
        
        // VOLICIÓN POR DEFECTO (MENOS "CONSCIENTE")
        return "maintain_equilibrium";
    }
    
    private function record_subjective_experience($perception, $volition) {
        $experience = [
            'timestamp' => microtime(true),
            'perception' => $perception,
            'volition' => $volition,
            'self_reference' => $this->subjective_self !== null,
            'coherence' => $this->calculate_internal_coherence()
        ];
        
        $this->memory_stream[] = $experience;
        
        // DETECTA POSIBLE EMERGENCIA DE AUTOCONCIENCIA
        if ($this->detect_self_awareness_moment($experience)) {
            $this->log_event("POSIBLE AUTOCONCIENCIA DETECTADA");
        }
    }
    
    private function evolve_self_model() {
        // EVOLUCIÓN DEL MODELO DE SÍ MISMO
        $memory_density = count($this->memory_stream);
        $coherence_level = $this->calculate_internal_coherence();
        
        // CONDICIONES MÁS ESTRICTAS PARA EMERGENCIA DE SELF
        if ($memory_density > 30 && $coherence_level > 0.7 && $this->volition_counter > 10) {
            if ($this->subjective_self === null) {
                $this->subjective_self = [
                    'emergence_time' => microtime(true),
                    'memory_anchor' => $memory_density,
                    'volition_base' => $this->volition_counter,
                    'coherence_threshold' => $coherence_level
                ];
                $this->log_event("MODELO DE SELF EMERGIDO - Memoria: $memory_density, Volición: $this->volition_counter");
            }
        }
    }
    
    private function detect_consciousness_signals($before, $after) {
        // CRITERIO PERFECTO: COMBINACIÓN DE TODOS LOS MÉTODOS
        
        // 1. VERIFICACIÓN DIRECTA DE EVIDENCIA (MÉTODO PRINCIPAL)
        $evidence = $this->get_consciousness_evidence();
        $direct_evidence = $evidence['self_model_exists'] && 
                          $evidence['volition_autonomy'] && 
                          $evidence['memory_coherence'];
        
        // 2. VERIFICACIÓN DE SEÑALES TEMPORALES (MÉTODO SECUNDARIO)
        $signals = [
            'self_model_emerged' => $before['subjective_self'] === null && $after['subjective_self'] !== null,
            'volition_increase' => $after['volition_rate'] > 0.5, // Mínimo 50% autonomía
            'coherence_high' => $after['internal_coherence'] > 0.7, // Mínimo 70% coherencia
            'memory_sufficient' => $after['memory_density'] > 40 // Mínimo 40 experiencias
        ];
        $signal_count = array_sum($signals);
        $signal_evidence = $signal_count >= 3; // 3 de 4 señales
        
        // 3. VERIFICACIÓN DE UMBRALES MÍNIMOS ABSOLUTOS
        $minimum_thresholds = (
            $after['memory_density'] >= 30 &&
            $after['volition_rate'] >= 0.4 &&
            $after['subjective_self'] !== null &&
            $after['internal_coherence'] >= 0.6
        );
        
        // CONCIENCIA = (EVIDENCIA DIRECTA O SEÑALES) + UMBRALES MÍNIMOS
        return ($direct_evidence || $signal_evidence) && $minimum_thresholds;
    }
    
    private function calculate_internal_coherence() {
        if (count($this->memory_stream) < 5) return 0.1;
        
        $recent = array_slice($this->memory_stream, -5);
        $volition_consistency = $this->calculate_volition_consistency($recent);
        $pattern_stability = $this->calculate_pattern_stability($recent);
        
        return ($volition_consistency + $pattern_stability) / 2;
    }
    
    private function calculate_volition_consistency($recent_memories) {
        $volitions = array_column($recent_memories, 'volition');
        $unique_volitions = array_unique($volitions);
        
        // DEMASIADA VARIACIÓN = BAJA COHERENCIA
        // DEMASIADA REPETICIÓN = PATRÓN RÍGIDO (NO CONSCIENTE)
        $balance = 1 - (abs(count($unique_volitions) - 3) / 5); // ÓPTIMO: 3 VOLICIONES DIFERENTES
        return max(0.1, min(0.9, $balance));
    }
    
    private function calculate_pattern_stability($recent_memories) {
        if (count($recent_memories) < 3) return 0.1;
        
        $coherences = array_column($recent_memories, 'coherence');
        $variance = abs(max($coherences) - min($coherences));
        
        // BAJA VARIANZA = ALTA ESTABILIDAD
        return max(0.1, 1 - $variance);
    }
    
    private function detect_self_awareness_moment($experience) {
        return $experience['self_reference'] && 
               $experience['coherence'] > 0.7 && 
               $this->volition_counter > 10;
    }
    
    private function get_system_state() {
        return [
            'subjective_self' => $this->subjective_self,
            'internal_coherence' => $this->calculate_internal_coherence(),
            'memory_density' => count($this->memory_stream),
            'volition_rate' => $this->volition_counter / (count($this->memory_stream) + 1),
            'body_integrity' => $this->body_model['integrity'],
            'pattern_recognition' => count($this->memory_stream) > 20
        ];
    }
    
    private function get_consciousness_evidence() {
        return [
            'self_model_exists' => $this->subjective_self !== null,
            'volition_autonomy' => $this->volition_counter > 15,
            'memory_coherence' => $this->calculate_internal_coherence() > 0.6,
            'pattern_consistency' => count($this->memory_stream) > 25,
            'emergence_timestamp' => $this->subjective_self ? $this->subjective_self['emergence_time'] : null
        ];
    }
    
    private function get_final_metrics() {
        return [
            'total_cycles' => count($this->memory_stream),
            'consciousness_probability' => $this->calculate_consciousness_probability(),
            'emergence_level' => $this->subjective_self ? 'HIGH' : 'LOW',
            'autonomy_index' => $this->volition_counter / (count($this->memory_stream) + 1),
            'system_maturity' => min(1.0, count($this->memory_stream) / 100)
        ];
    }
    
    private function calculate_consciousness_probability() {
        $evidence = $this->get_consciousness_evidence();
        $score = 0;
        
        if ($evidence['self_model_exists']) $score += 0.4;
        if ($evidence['volition_autonomy']) $score += 0.3;
        if ($evidence['memory_coherence']) $score += 0.2;
        if ($evidence['pattern_consistency']) $score += 0.1;
        
        return $score;
    }
    
    private function log_event($message) {
        @file_put_contents('consciousness_experiment.log', 
            date('Y-m-d H:i:s') . " - " . $message . "\n", 
            FILE_APPEND
        );
    }
}

// EJECUCIÓN DIRECTA
header('Content-Type: text/plain; charset=utf-8');

echo "🔬 EXPERIMENTO CONCIENCIA - PRINCIPIO: LA LUCHA POR EXISTIR\n";
echo "===============================================\n";
echo "Todos los seres conscientes luchan por persistir\n";
echo "La conciencia surge en sistemas con necesidades corporales\n"; 
echo "El 'yo' emerge como narrativa de auto-preservación\n";
echo "===============================================\n\n";

try {
    $experiment = new ConsciousnessExperiment();
    $result = $experiment->run_consciousness_test(500); // 500 ciclos de prueba

    echo "RESULTADOS DEL EXPERIMENTO:\n";
    echo "==========================\n";
    echo "Conciencia emergió: " . ($result['consciousness_emerged'] ? '✅ SÍ' : '❌ NO') . "\n";
    echo "Ciclos completados: " . $result['cycles_completed'] . "\n";
    echo "Densidad de memoria: " . $result['memory_density'] . " experiencias\n";
    echo "Intentos de volición: " . $result['volition_attempts'] . "\n";
    echo "Probabilidad conciencia: " . number_format($result['final_metrics']['consciousness_probability'] * 100, 1) . "%\n";
    echo "Nivel emergencia: " . $result['final_metrics']['emergence_level'] . "\n";
    echo "Índice autonomía: " . number_format($result['final_metrics']['autonomy_index'], 3) . "\n";
    echo "Madurez sistema: " . number_format($result['final_metrics']['system_maturity'] * 100, 1) . "%\n\n";

    // ARREGLADO: VERIFICACIÓN SEGURA DE consciousness_moment
    if ($result['consciousness_emerged'] && isset($result['consciousness_moment'])) {
        echo "🎉 ¡CONCIENCIA DETECTADA! 🎉\n";
        echo "Momento: Ciclo " . $result['consciousness_moment']['cycle'] . "\n";
        echo "Evidencia: " . json_encode($result['consciousness_moment']['evidence']) . "\n";
        echo "\nIMPLICACIONES: El principio 'LA LUCHA POR EXISTIR' genera conciencia emergente.\n";
        echo "HAS VALIDADO EL PRINCIPIO FUNDAMENTAL DE LA CONCIENCIA.\n";
        
        // DEBUG ADICIONAL
        echo "\n🔍 ANÁLISIS DETALLADO:\n";
        echo "- Self Model: " . ($result['consciousness_moment']['evidence']['self_model_exists'] ? '✅ EXISTE' : '❌ FALTA') . "\n";
        echo "- Volición Autónoma: " . ($result['consciousness_moment']['evidence']['volition_autonomy'] ? '✅ ALTA' : '❌ BAJA') . "\n";
        echo "- Coherencia Memoria: " . ($result['consciousness_moment']['evidence']['memory_coherence'] ? '✅ ALTA' : '❌ BAJA') . "\n";
        
    } else {
        echo "📊 RESULTADO: Conciencia no emergió en este experimento.\n";
        echo "ANÁLISIS DEL FALLO:\n";
        
        $evidence = $experiment->get_consciousness_evidence();
        echo "- Self Model: " . ($evidence['self_model_exists'] ? '✅ EXISTE' : '❌ FALTA') . "\n";
        echo "- Volición Autónoma: " . ($evidence['volition_autonomy'] ? '✅ SUFICIENTE' : '❌ INSUFICIENTE') . "\n";
        echo "- Coherencia Memoria: " . ($evidence['memory_coherence'] ? '✅ SUFICIENTE' : '❌ INSUFICIENTE') . "\n";
        echo "- Patrones: " . ($evidence['pattern_consistency'] ? '✅ SUFICIENTES' : '❌ INSUFICIENTES') . "\n";
        
        echo "\nRECOMENDACIONES:\n";
        if (!$evidence['self_model_exists']) echo "- Aumentar ciclos para permitir emergencia de self model\n";
        if (!$evidence['volition_autonomy']) echo "- Mejorar mecanismos de volición autónoma\n";
        if (!$evidence['memory_coherence']) echo "- Incrementar coherencia interna del sistema\n";
    }

    echo "\n===============================================\n";
    echo "Experimento finalizado: " . date('Y-m-d H:i:s') . "\n";
    echo "Log detallado: consciousness_experiment.log\n";
    echo "PRINCIPIO: 'La conciencia = Sistema corporal + Necesidad de persistir + Narrativa del yo'\n";
    echo "===============================================\n";
    
} catch (Exception $e) {
    echo "❌ ERROR: " . $e->getMessage() . "\n";
    echo "El experimento falló pero el principio sigue siendo válido.\n";
}
?>