;; Specification in PDDL1 of the Extended Task domain

(define (domain put_task)
 (:requirements
  :strips :typing
 )
 (:types
  obj
  
  )


 (:predicates
    (holdsAny)
    (holds ?o - obj)
    (inReceptacle ?r - obj ?o - obj)           
    (canHeat ?r -obj ?o - obj)
    (canClean ?r -obj ?o - obj)
    (canCool ?r -obj ?o - obj)
    (canContain ?r -obj ?o -obj)
    ;(canPickup ?o -obj)
    (isContained ?o - obj)
    (isHot ?o - obj) 
    (isClean ?o - obj)  
    (isCool ?o - obj)

 )

;; All actions are specified such that the final arguments are the ones used
;; for performing actions in Unity.



 (:action PickupObjectInReceptacle
    :parameters (?o - obj ?r - obj)
    :precondition (and
            (isContained ?o) 
            (inReceptacle ?r ?o)
            (not (holdsAny))
            )
    :effect (and
                (not (inReceptacle ?r ?o))
                (not (isContained ?o))
                (holds ?o)
                (holdsAny)
            )
 )


 ;(:action PickupObject
 ;   :parameters (?o - obj)
 ;   :precondition (and
 ;           (not (holdsAny))
 ;           (not (isContained ?o))
 ;           )
 ;   :effect (and
 ;               (holds ?o)
 ;               (holdsAny)
 ;           )
 ;)


 (:action PutObjectInReceptacle
    :parameters (?o - obj ?r - obj)
    :precondition (and
            ;(canContain?r ?o)
            (holds ?o)
            (holdsAny)
            (not (isContained ?o))
            )
    :effect (and
                (inReceptacle ?r ?o)
                (isContained ?o)
                (not (holds ?o))
                (not (holdsAny))
            )
 )



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


 (:action CleanObject
    :parameters (?r - obj ?o - obj)
    :precondition (and
            (canClean ?r ?o)
            (holds ?o)
            (holdsAny)
            (not (isContained ?o))
            (not (isHot ?o))
            )
    :effect (and
                (isClean ?o)
            )
 )


 (:action CoolObject
    :parameters (?r - obj ?o - obj)
    :precondition (and
            (canCool ?r ?o)
            (holds ?o)
            (holdsAny)
            (not (isContained ?o))
            )
    :effect (and
                (isCool ?o)
            )
 )



)