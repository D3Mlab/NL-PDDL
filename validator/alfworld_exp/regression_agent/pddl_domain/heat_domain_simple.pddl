;; Specification in PDDL1 of the Extended Task domain

(define (domain put_task)
 (:requirements
  :strips :typing
 )
 (:types
  obj
  
  )


 (:predicates

    ;(atObject ?o - obj)
    (holdsAny)
    (holds ?o - obj)
    ;(objectAtLocation ?o - obj ?l - obj)              ; true if the object is at the location
    (inReceptacle ?o - obj ?r - obj)                ; object ?o is in receptacle ?r
    ;(holds ?o - obj)                            ; object ?o is held by agent ?a
    (isHot ?o - obj)                                       ; true if the object has been heated up
    (canHeat ?r -obj ?o - obj)
    (isPotato ?o - obj)
    (isPlate ?o - obj)
    (isContained ?o - obj)  

 )

;; All actions are specified such that the final arguments are the ones used
;; for performing actions in Unity.


;; agent picks up object
 (:action PickupObjectInReceptacle
    :parameters (?o - obj ?r - obj)
    :precondition (and
            (isContained ?o) 
            (inReceptacle ?o ?r)
            (not (holdsAny))
            )
    :effect (and
                (not (inReceptacle ?o ?r))
                (not (isContained ?o))
                (holds ?o)
                (holdsAny)
            )
 )

;; agent picks up object not in a receptacle
 (:action PickupObject
    :parameters (?o - obj)
    :precondition (and
            (not (holdsAny))
            (not (isContained ?o))
            )
    :effect (and
                (holds ?o)
                (holdsAny)
            )
 )

;; agent puts down an object in a receptacle
 (:action PutObjectInReceptacle
    :parameters (?o - obj ?r - obj)
    :precondition (and
            (holds ?o)
            (holdsAny)
            (not (isContained ?o))
            )
    :effect (and
                (inReceptacle ?o ?r)
                (isContained ?o)
                (not (holds ?o))
                (not (holdsAny))
            )
 )


;; agent heats-up some object
 (:action HeatObject
    :parameters (?r - obj ?o - obj)
    :precondition (and
            (canHeat ?r ?o)
            (holds ?o)
            (holdsAny)
            (not (isContained ?o))
            )
    :effect (and
                (isHot ?o)
            )
 )


)