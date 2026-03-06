The Marketplace must restrict agent maintenance and management actions to authorized users only.

When a user accesses the Marketplace or attempts to manage an agent, the system will perform an Active Directory (AD) lookup using APS APIs to determine whether the user belongs to the configured team distribution list (DL) associated with the agent.

If the user is part of the team DL, the system will mark them as authorized (isOwner = true) and allow access to agent management capabilities.

If the user is not part of the team DL, the system will mark them as unauthorized (isOwner = false) and restrict access to those actions.

This authorization logic will be used by the Marketplace agent management and playground layers.



AC1 — AD Lookup Validation

When a user logs into the Marketplace, the system must perform an AD lookup using APS endpoints with the user's AskID.

AC2 — Authorization Determination

The system must evaluate whether the user belongs to the agent's configured team distribution list.

AC3 — Boolean Authorization Flag

The system must return a boolean flag:

isOwner = true  → user is part of team DL
isOwner = false → user is not part of team DL
AC4 — Agent Access Control

If isOwner = true, the user may:

manage agents

update agents

access maintenance features

If isOwner = false, the user may:

view agents only

not modify agent configuration

AC5 — API Integration

The Marketplace backend must integrate with APS AD Lookup APIs using the client credential flow.

AC6 — Error Handling

If the AD lookup API fails:

system logs error

user access defaults to unauthorized
