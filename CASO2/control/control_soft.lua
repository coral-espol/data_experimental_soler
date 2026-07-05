-- Gabriel Mauricio Madroñero Pachajoa
-- Proyecto: Evaluación de costos y beneficios del aprendizaje en enjambres de robots.
------------------------ 1. PARÁMETROS DE APRENDIZAJE ------------------------
local T_TICKS_PER_SEC = 10.0

-- VARIABLES GLOBALES para el C++ Loop Functions
m = 0.0             -- Memoria de aprendizaje (m)
p_x = 0.0           -- Probabilidad de selección de tarea
planned_wticks = 0  -- Tiempo de servicio 
search_ticks = 0    -- Tiempo gastado buscando tarea

-- VARIABLES MODELO APRENDIZAJE
GAMMA = 0.2 
local N_MAX = 12.0
local W_STD = 60--120.0
local K_GAIN = 1.15--1.1538--1.25 
local CROSS_FORGET = 1.0
local DF_DECAY_DIST = 300.0

------------------------ APRENDIZAJE SOCIAL ------------------------

-- constante de aprendizaje social
ALPHA_SOCIAL = 0.9   -- valores a probar: 0.0, 0.3, 0.6, 0.9
-- Radio de interacción social en cm 
local SOCIAL_LEARN_RADIUS_CM = 60.0
-- contante de penalizacion social 
BETA_SOCIAL = 0.8 -- valores a probar: 0.0, 0.2, 0.5, 0.8
SOCIAL_BOOL = true -- variables que itera entre dos estrategias de conteo de robots: true =conteo iterativo, false snapshot  unico al inicio.
local snapshot_count = 0  -- Para la función Snapshot
local max_social_seen = 0 -- Para la función Iterativa

------------------------ 2. PARÁMETROS NAVEGACIÓN ------------------------
local BASE_SPEED = 7.0
local TURN_GAIN = 2.0
local THRESHOLD = 0.1
local VMAX = 8.0

-- FASES Y TIEMPOS
local EVALUATION_TIME = 3  -- Ticks para estabilizar lectura de suelo

-- CONFIG GLOBAL
GREEDY_MODE = true -- Estrategia: true = Greedy, false = Selective
local CM_PER_POS_UNIT = 100.0

-- Radio de ocupación social (cm)
local OCCUPATION_RADIUS_CM = 60.0
local OCCUPATION_ANGLE_RAD = math.rad(40)
local OCCUPATION_CONFIRM_T = 1
local occupation_counter = 0

-- Evasión anticipativa
local SOCIAL_AVOID_RADIUS_CM = 60.0
local SOCIAL_TURN_GAIN = 6.0
local SOCIAL_SLOW_FACTOR = 0.4

-- Evasión Globales
local avoid_counter = 0
local avoid_vL = 0
local avoid_vR = 0
local AVOID_TICKS = 1

------------------------ 3. VARIABLES DE ESTADO INTERNAS ------------------------
STATE = "EXPLORE"
local current_target_type = "NONE" 
local learn_count_b = 0.0 -- Tarea Azul (White)
local learn_count_g = 0.0 -- Tarea Roja (Black)
local last_position = {x=0.0, y=0.0}
local accumulated_dist = 0.0
------------------------ 4. INFRAESTRUCTURA Y SENSORES ------------------------

local function clamp(x, min, max) return math.max(min, math.min(max, x)) end

local function apply_and_return(vL, vR)
  vL = clamp(vL, -VMAX, VMAX)
  vR = clamp(vR, -VMAX, VMAX)
  robot.wheels.set_velocity(vL, vR)
  return true
end

local function set_led_color(color)
  if robot.leds and robot.leds.set_all_colors then
    if color == "RED" then robot.leds.set_all_colors(255,0,0)
    elseif color == "BLUE" then robot.leds.set_all_colors(0,0,255)
    else robot.leds.set_all_colors(0,0,0) end
  end
end

--------------------------------------------------
-- FUNCIONES DE CÁMARA 
--------------------------------------------------
local function get_unique_robots()
  local cam = robot.colored_blob_omnidirectional_camera
  local readings = cam.readings or cam

  -- Cámara activada desde EXPLORE para máxima trazabilidad
  if STATE == "EXPLORE" or STATE == "EVALUATE" or STATE == "ENTER_PATCH" or STATE == "EXECUTE" then
    if readings and #readings > 0 then
      log("======================".. robot.id .."----"..STATE.."======================") 
      log("[" .. robot.id .. "] VE " .. #readings .. " BLOBS:")
      for i, blob in ipairs(readings) do
        log("  -> Color (RGB): " .. blob.color.red .. "," .. blob.color.green .. "," .. blob.color.blue .. " | Dist: " .. blob.distance)
      end
    -- else
      -- log("[" .. robot.id .. "] ESTÁ TOTALMENTE CIEGO (0 blobs)")
    end
  end

  if not readings then return {} end

  local valid_blobs = {}
  
  -- LÓGICA SIMPLIFICADA: Cero agrupación. Pasamos el conteo directo de blobs.
  for _, blob in ipairs(readings) do
    local r, g, b = blob.color.red, blob.color.green, blob.color.blue
    
    local color_key = "NONE"
    if r == 255 and g == 0 and b == 0 then color_key = "RED"
    elseif r == 0 and g == 0 and b == 255 then color_key = "BLUE" end

    if color_key ~= "NONE" then
      -- Se inserta directamente cada blob válido
      table.insert(valid_blobs, {color = color_key, dist = blob.distance, angle = blob.angle})
    end
  end
  
  if #valid_blobs > 0 then
      log("----- # BLOBS VALIDOS ="..#valid_blobs.. "-----")
  end
  
  return valid_blobs
end

local function cam_blobs()
  local cam = robot.colored_blob_omnidirectional_camera
  if not cam then return {} end
  return cam.readings or cam
end

local function social_avoidance_velocity(base_vL, base_vR)

  for _, blob in ipairs(cam_blobs()) do
    if blob.color and blob.distance and blob.angle then

      local r = blob.color.red or 0
      local g = blob.color.green or 0
      local b = blob.color.blue or 0

      local is_exec_robot =
        (r == 255 and g == 0 and b == 0) or
        (r == 0 and g == 0 and b == 255)

      if is_exec_robot then

        local dist_cm = blob.distance * CM_PER_POS_UNIT

        if dist_cm <= SOCIAL_AVOID_RADIUS_CM then

          local slow = SOCIAL_SLOW_FACTOR

          if blob.angle > 0 then
            return base_vL * slow + SOCIAL_TURN_GAIN,
                   base_vR * slow - SOCIAL_TURN_GAIN
          else
            return base_vL * slow - SOCIAL_TURN_GAIN,
                   base_vR * slow + SOCIAL_TURN_GAIN
          end

        end
      end
    end
  end

  return base_vL, base_vR
end

--------------------------------------------------
-- CONTEO SOCIAL DE VECINOS
--------------------------------------------------

local function count_neighbors_same_task()
  local count = 0
  -- Ahora recibe la lista de blobs crudos en lugar de robots agrupados
  local blobs = get_unique_robots() 
  for _, res in ipairs(blobs) do
    if res.dist <= SOCIAL_LEARN_RADIUS_CM and res.color == current_target_type then
      count = count + 1
    end
  end
  if count > 0 then
      log("----- # BLOBS RETORNO PARA LA RECO ="..count.. "-----")
  end
  return count
end

local function count_neighbors_different_task()
  local count = 0
  -- Ahora recibe la lista de blobs crudos en lugar de robots agrupados
  local blobs = get_unique_robots() 
  for _, res in ipairs(blobs) do
    if res.dist <= SOCIAL_LEARN_RADIUS_CM and res.color ~= current_target_type then
      count = count + 1
    end
  end
  if count > 0 then
      log("----- # BLOBS RETORNO PARA PENALIZACION ="..count.. "-----")
  end
  return count
end

local function patch_is_free()
  local blobs = get_unique_robots()
  local occupied_detected = false

  for _, res in ipairs(blobs) do
    if res.dist <= OCCUPATION_RADIUS_CM and math.abs(res.angle) <= OCCUPATION_ANGLE_RAD then
      occupied_detected = true
      break
    end
  end

  occupation_counter = occupied_detected and (occupation_counter + 1) or 0
  return occupation_counter < OCCUPATION_CONFIRM_T
end
--------------------------------------------------

local function ground_avg_gray()
  local g = robot.qupa_ground or robot.ground or robot.base_ground or robot.motor_ground
  if not g then return nil end
  local readings = g.readings or g
  local v = nil
  if type(readings) == "table" and #readings > 0 then
    local r1 = readings[1]
    if type(r1) == "number" then v = r1
    elseif type(r1) == "table" then
      v = r1.value or r1.gray
    end
  elseif type(readings) == "number" then v = readings end
  if v and v > 2.0 then v = v / 255.0 end
  return v
end

local function get_ground_task()
  local avg = ground_avg_gray()
  if avg == nil then return "NONE" end
  if avg < 0.1 then return "RED" end
  if avg > 0.9 then return "BLUE" end
  return "NONE"
end

------------------------ 5. MATEMÁTICAS DEL MODELO ------------------------
local function calculate_prob_accept(target_type)
  log("------ Variable m ".. m)
  local p_red = 1.0 / (1.0 + math.exp(-GAMMA * m))
  if target_type == "RED" then return p_red end
  if target_type == "BLUE" then return (1.0 - p_red) end
  return 0.0
end

local function calculate_service_time(n_count)
  if n_count <= 0 then return math.ceil(W_STD * T_TICKS_PER_SEC) end
  local c = N_MAX / 2.0
  local sigmoid = 1.0 / (1.0 + math.exp(-(n_count - c)))
  local time_saved = (W_STD / K_GAIN) * sigmoid
  return math.ceil((W_STD - time_saved) * T_TICKS_PER_SEC)
end

local function apply_distance_decay()
  if not robot.position then return end
  local d = math.sqrt((robot.position.x - last_position.x)^2 + (robot.position.y - last_position.y)^2)
  accumulated_dist = accumulated_dist + (d * CM_PER_POS_UNIT)
  last_position = {x=robot.position.x, y=robot.position.y}

  while accumulated_dist >= DF_DECAY_DIST do
    accumulated_dist = accumulated_dist - DF_DECAY_DIST
    learn_count_b = math.max(0, learn_count_b - 1)
    learn_count_g = math.max(0, learn_count_g - 1)
    if m > 0 then m = math.max(0, m - 1)
    elseif m < 0 then m = math.min(0, m + 1) end
  end
end

------------------------ 6. NAVEGACIÓN Y EVASIÓN ------------------------
local function drive_avoidance()
    if STATE == "ENTERING" or STATE == "EXECUTE" then
        return BASE_SPEED, BASE_SPEED, 0.0
    end
    
    if avoid_counter > 0 then
        avoid_counter = avoid_counter - 1
        return avoid_vL, avoid_vR, 1.0
    end

    local prox = robot.qupa_proximity or robot.proximity
    local readings = prox and (prox.readings or prox) or {}
    
    local val_f  = readings[1] and readings[1].value or 0
    local val_fr = readings[2] and readings[2].value or 0
    local val_fl = readings[3] and readings[3].value or 0
    local val_r  = readings[4] and readings[4].value or 0
    local val_l  = readings[5] and readings[5].value or 0

    local mag = math.max(val_f, val_fr, val_fl, val_r, val_l)
    local speed_factor = math.max(0.1, 1.0 - mag)
    local current_base_speed = BASE_SPEED * speed_factor

    if mag > THRESHOLD then
        local vL, vR = current_base_speed, current_base_speed
        
        if val_fr > val_fl or val_r > val_l then
            vL = current_base_speed - TURN_GAIN
            vR = current_base_speed + TURN_GAIN
        else
            vL = current_base_speed + TURN_GAIN
            vR = current_base_speed - TURN_GAIN
        end

        if val_f > 0.7 and math.abs(val_fr - val_fl) < 0.1 then
            vL = VMAX
            vR = -VMAX
            avoid_counter = AVOID_TICKS * 2
        else
            avoid_counter = AVOID_TICKS
        end

        avoid_vL = clamp(vL, -VMAX, VMAX)
        avoid_vR = clamp(vR, -VMAX, VMAX)
        return avoid_vL, avoid_vR, mag
    end

    return BASE_SPEED, BASE_SPEED, 0.0
end

------------------------ 7. MÁQUINA DE ESTADOS ------------------------

local execute_ticks = 0
local cooldown_ticks = 0
local evaluation_counter = 0
local stable_task = "NONE"

local exit_counter = 0
local EXIT_CONFIRM_T = 1

function step()

  apply_distance_decay()

  --------------------------------------------------
  -- FSM
  --------------------------------------------------

  if STATE == "EXPLORE" then

    search_ticks = search_ticks + 1
    
    -- Giro simétrico puro
    apply_and_return(-4, 4)

    task = get_ground_task()

    if task ~= "NONE" then
      -- Arrancamos la evaluación
      evaluation_counter = 1
      stable_task = task
      STATE = "EVALUATE"
    end
    log("---------------------.EXPLORANDO----------------------------- "..robot.id)
  --------------------------------------------------

  elseif STATE == "EVALUATE" then
    log("---------------------INICIO ESTADO EVALUATE----------------------------- "..robot.id)
    -- Aplicamos giro simétrico
    apply_and_return(0, 0)

    -- Comentamos la restricción de ocupación para la prueba estática
    -- if not patch_is_free() then
    --   STATE = "EXPLORE"
    --   return
    -- end

    local task = get_ground_task()

    if task == stable_task then
      evaluation_counter = evaluation_counter + 1
    else
      STATE = "EXPLORE"
      return
    end

    -- Evaluamos en solo 1 tick (0.1s) 
    if evaluation_counter >= 1 then

      current_target_type = stable_task
      local accept = false

      if GREEDY_MODE == true then
        accept = true
        p_x = 1.0
      else
        local p = calculate_prob_accept(stable_task)
        p_x = p
        local rand_val = robot.random.uniform(0.0,1)
        log("----- Random value = "..rand_val.. " vs p = "..p.. " -----")
        if rand_val <= p then -- Generador aleatorio de numeros revizar el azar de la evaluacion
          accept = true
        end
      end

      if accept then
        STATE = "ENTER_PATCH"
      else
        STATE = "EXPLORE"
      end
      log("----- ESTADO = ----"..STATE.. "----- "..robot.id)
    end

  --------------------------------------------------

  elseif STATE == "ENTER_PATCH" then
    log("---------------------INICIO ESTADO ENTER_PATCH----------------------------- "..robot.id)
    -- if not patch_is_free() then
    --   STATE = "EXPLORE"
    --   return
    -- end

    local task = get_ground_task()

    if task ~= "NONE" then
    --  exit_counter = 0
    --  apply_and_return(-4, 4)
    --else
      exit_counter = exit_counter + 1
      if exit_counter >= EXIT_CONFIRM_T then
        apply_and_return(0,0)
        execute_ticks = calculate_service_time(
          (current_target_type=="RED" and learn_count_g or learn_count_b)
        )
        planned_wticks = execute_ticks
        snapshot_count = count_neighbors_same_task()
        max_social_seen = 0

        STATE = "EXECUTE"
      end
      log("----- ESTADO ENTER_PATCH - Tiempo de servicio "..planned_wticks.. "-----"..robot.id)
    end

  --------------------------------------------------

  elseif STATE == "EXECUTE" then
    log("---------------------INICIO ESTADO EXECUTE----------------------------- "..robot.id)
    apply_and_return(0, 0)
    
    set_led_color(current_target_type)
    
    local neighbors = count_neighbors_same_task()
    local oposite_neighbors = count_neighbors_different_task()
    if neighbors > max_social_seen then
       max_social_seen = neighbors
    end

    execute_ticks = execute_ticks - 1

    if execute_ticks <= 0 then
      local n_effective = 0
      if SOCIAL_BOOL == true then
        n_effective = max_social_seen 
      else
        n_effective = snapshot_count 
      end
      log("[" ..n_effective.. "] calculo de recompensa")
      log("[" ..oposite_neighbors.. "] calculo de penalizacion")
      
      local reward = 0.0
      if n_effective < 3 then
          reward = 1.0 + (ALPHA_SOCIAL * n_effective)
      else
          reward = 3.7
      end
      
      local delta = reward 

      local base_penality = 0.0
      if oposite_neighbors < 2 then
          base_penality = 1.0 + (1.5 * oposite_neighbors)
      else
          base_penality = 3.7
      end
      
      local penality = CROSS_FORGET * base_penality

      log("Penalizacion= "..penality.." Recompensa= "..delta)
      if current_target_type == "RED" then
        learn_count_g = learn_count_g + delta
        learn_count_b = math.max(0, learn_count_b - penality)
        m = clamp(m + delta, -N_MAX, N_MAX)
      else
        learn_count_b = learn_count_b + delta
        learn_count_g = math.max(0, learn_count_g - penality)
        m = clamp(m - delta, -N_MAX, N_MAX)
      end
      log("contador blue "..learn_count_b.." contador red= "..learn_count_g)
      -- Damos 30 ticks 3 segundos de tiempo muerto visible
      cooldown_ticks = 30
      set_led_color("NONE")
      STATE = "COOLDOWN"
    end
  --------------------------------------------------
  elseif STATE == "COOLDOWN" then
    log("----- ESTADO COOLDOWN -----"..STATE.. "-----" ..robot.id )
    apply_and_return(0, 0)
    cooldown_ticks = cooldown_ticks - 1
    log("----- TIEMPO MUERTO "..cooldown_ticks.. "-----")
    if cooldown_ticks <= 0 then
      search_ticks = 0
      STATE = "EXPLORE"
    end

  end
end

function init()
  STATE = "EXPLORE"
  m = 0
  learn_count_b = 0
  learn_count_g = 0
  accumulated_dist = 0
  planned_wticks = 0
  search_ticks = 0

  if robot.colored_blob_omnidirectional_camera then
    robot.colored_blob_omnidirectional_camera.enable()
  end

  if robot.position then
    last_position = {x=robot.position.x, y=robot.position.y}
    wd_last_pos = {x=robot.position.x, y=robot.position.y}
  else
    last_position = {x=0, y=0}
    wd_last_pos = {x=0, y=0}
  end
end

function reset()
  init()
end

function destroy()
  --
end